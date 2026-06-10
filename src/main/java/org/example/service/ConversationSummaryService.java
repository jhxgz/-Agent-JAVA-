package org.example.service;

import com.alibaba.cloud.ai.dashscope.api.DashScopeApi;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.graph.agent.ReactAgent;
import com.alibaba.cloud.ai.graph.exception.GraphRunnerException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * 总结对话 Agent：将多轮历史对话压缩为摘要，避免上下文窗口溢出。
 */
@Service
public class ConversationSummaryService {

    private static final Logger logger = LoggerFactory.getLogger(ConversationSummaryService.class);

    public static final String SUMMARY_ROLE = "summary";

    private static final String SUMMARY_AGENT_SYSTEM_PROMPT = """
            你是「总结对话 Agent」，负责将多轮对话压缩为简洁摘要。
            要求：
            1. 保留用户的核心问题、已确认的事实、工具查询结论与最终处理方案。
            2. 使用第三人称客观描述，2～5 句话，不超过 300 字。
            3. 不要编造未出现的信息，不要输出 JSON 或 Markdown 标题。
            """;

    @Autowired
    private DashScopeModelFactory modelFactory;

    /**
     * 对指定轮次对话生成摘要；若已有历史摘要，则合并进新的摘要中。
     *
     * @param dialoguePairs 待压缩的对话消息（user/assistant 交替）
     * @param previousSummary 已有摘要，可为 null
     * @return 压缩后的摘要文本
     */
    public String summarize(List<Map<String, String>> dialoguePairs, String previousSummary) {
        String dialogueText = formatDialogue(dialoguePairs);
        String userInput = buildSummaryInput(dialogueText, previousSummary);

        try {
            DashScopeApi dashScopeApi = modelFactory.createDashScopeApi();
            DashScopeChatModel summaryModel = modelFactory.createSummaryChatModel(dashScopeApi);

            ReactAgent summaryAgent = ReactAgent.builder()
                    .name("conversation_summary_agent")
                    .model(summaryModel)
                    .systemPrompt(SUMMARY_AGENT_SYSTEM_PROMPT)
                    .build();

            logger.info("调用总结对话 Agent，待压缩消息条数: {}", dialoguePairs.size());
            String summary = summaryAgent.call(userInput).getText();
            logger.info("对话摘要生成完成，长度: {}", summary != null ? summary.length() : 0);
            return summary != null ? summary.trim() : "";
        } catch (GraphRunnerException e) {
            logger.error("总结对话 Agent 调用失败，回退为截断摘要", e);
            return fallbackSummary(dialoguePairs, previousSummary);
        }
    }

    private String buildSummaryInput(String dialogueText, String previousSummary) {
        StringBuilder input = new StringBuilder();
        if (previousSummary != null && !previousSummary.isBlank()) {
            input.append("【已有历史摘要】\n").append(previousSummary).append("\n\n");
        }
        input.append("【待压缩的对话内容】\n").append(dialogueText);
        input.append("\n\n请输出合并后的最新对话摘要。");
        return input.toString();
    }

    private String formatDialogue(List<Map<String, String>> messages) {
        StringBuilder builder = new StringBuilder();
        for (Map<String, String> msg : messages) {
            String role = msg.get("role");
            String content = msg.get("content");
            if ("user".equals(role)) {
                builder.append("用户: ").append(content).append("\n");
            } else if ("assistant".equals(role)) {
                builder.append("助手: ").append(content).append("\n");
            }
        }
        return builder.toString();
    }

    private String fallbackSummary(List<Map<String, String>> dialoguePairs, String previousSummary) {
        StringBuilder fallback = new StringBuilder();
        if (previousSummary != null && !previousSummary.isBlank()) {
            fallback.append(previousSummary).append(" ");
        }
        for (Map<String, String> msg : dialoguePairs) {
            if ("user".equals(msg.get("role"))) {
                String content = msg.get("content");
                if (content != null && !content.isBlank()) {
                    fallback.append(content, 0, Math.min(content.length(), 80)).append("；");
                }
            }
        }
        return fallback.toString().trim();
    }
}
