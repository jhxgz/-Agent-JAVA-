package org.example.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 会话记忆管理：维护多轮对话历史，并在超过阈值时调用总结 Agent 压缩早期对话。
 */
@Service
public class ChatSessionService {

    private static final Logger logger = LoggerFactory.getLogger(ChatSessionService.class);

    @Autowired
    private ConversationSummaryService summaryService;

    @Value("${chat.memory.max-recent-pairs:5}")
    private int maxRecentPairs;

    @Value("${chat.memory.compress-batch-pairs:5}")
    private int compressBatchPairs;

    private final Map<String, SessionInfo> sessions = new ConcurrentHashMap<>();

    public SessionInfo getOrCreateSession(String sessionId) {
        if (sessionId == null || sessionId.isEmpty()) {
            sessionId = UUID.randomUUID().toString();
        }
        final String resolvedId = sessionId;
        return sessions.computeIfAbsent(resolvedId, SessionInfo::new);
    }

    public SessionInfo getSession(String sessionId) {
        return sessions.get(sessionId);
    }

    public void clearSession(String sessionId) {
        SessionInfo session = sessions.get(sessionId);
        if (session != null) {
            session.clearHistory();
        }
    }

    /**
     * 追加一轮对话，并在超过 maxRecentPairs 时压缩最早的 compressBatchPairs 轮。
     */
    public void addMessageWithCompression(String sessionId, String userQuestion, String aiAnswer) {
        SessionInfo session = getOrCreateSession(sessionId);
        session.addMessagePair(userQuestion, aiAnswer);
        session.compressHistoryIfNeeded(summaryService, maxRecentPairs, compressBatchPairs);
    }

    /**
     * 单个会话的历史与压缩逻辑。
     */
    public static class SessionInfo {

        private final String sessionId;
        private final List<Map<String, String>> messageHistory = new ArrayList<>();
        private final long createTime = System.currentTimeMillis();
        private final ReentrantLock lock = new ReentrantLock();

        public SessionInfo(String sessionId) {
            this.sessionId = sessionId;
        }

        public String getSessionId() {
            return sessionId;
        }

        public long getCreateTime() {
            return createTime;
        }

        public void addMessagePair(String userQuestion, String aiAnswer) {
            lock.lock();
            try {
                messageHistory.add(messageOf("user", userQuestion));
                messageHistory.add(messageOf("assistant", aiAnswer));
                logger.debug("会话 {} 追加一轮对话，当前原始轮数: {}", sessionId, getRawPairCount());
            } finally {
                lock.unlock();
            }
        }

        /**
         * 当原始对话轮数超过 maxRecentPairs 时，将最早的 compressBatchPairs 轮压缩为摘要。
         */
        void compressHistoryIfNeeded(ConversationSummaryService summaryService,
                                     int maxRecentPairs,
                                     int compressBatchPairs) {
            lock.lock();
            try {
                while (getRawPairCount() > maxRecentPairs) {
                    String previousSummary = extractSummaryContent();
                    List<Map<String, String>> pairsToCompress = extractFirstRawMessages(compressBatchPairs);
                    if (pairsToCompress.isEmpty()) {
                        break;
                    }

                    String newSummary = summaryService.summarize(pairsToCompress, previousSummary);
                    removeSummaryIfPresent();
                    removeFirstRawMessages(compressBatchPairs);
                    prependSummary(newSummary);

                    logger.info("会话 {} 已压缩 {} 轮对话为摘要，当前原始轮数: {}",
                            sessionId, compressBatchPairs, getRawPairCount());
                }
            } finally {
                lock.unlock();
            }
        }

        public List<Map<String, String>> getHistory() {
            lock.lock();
            try {
                return new ArrayList<>(messageHistory);
            } finally {
                lock.unlock();
            }
        }

        public void clearHistory() {
            lock.lock();
            try {
                messageHistory.clear();
                logger.info("会话 {} 历史消息已清空", sessionId);
            } finally {
                lock.unlock();
            }
        }

        public int getMessagePairCount() {
            lock.lock();
            try {
                return getRawPairCount() + (hasSummary() ? 1 : 0);
            } finally {
                lock.unlock();
            }
        }

        public int getRawPairCount() {
            int summaryOffset = hasSummary() ? 1 : 0;
            return (messageHistory.size() - summaryOffset) / 2;
        }

        public boolean hasSummary() {
            return !messageHistory.isEmpty()
                    && ConversationSummaryService.SUMMARY_ROLE.equals(messageHistory.get(0).get("role"));
        }

        private String extractSummaryContent() {
            if (!hasSummary()) {
                return null;
            }
            return messageHistory.get(0).get("content");
        }

        private List<Map<String, String>> extractFirstRawMessages(int pairCount) {
            int startIndex = hasSummary() ? 1 : 0;
            int endIndex = Math.min(startIndex + pairCount * 2, messageHistory.size());
            return new ArrayList<>(messageHistory.subList(startIndex, endIndex));
        }

        private void removeSummaryIfPresent() {
            if (hasSummary()) {
                messageHistory.remove(0);
            }
        }

        private void removeFirstRawMessages(int pairCount) {
            int startIndex = hasSummary() ? 1 : 0;
            int removeCount = Math.min(pairCount * 2, messageHistory.size() - startIndex);
            for (int i = 0; i < removeCount; i++) {
                messageHistory.remove(startIndex);
            }
        }

        private void prependSummary(String summaryContent) {
            Map<String, String> summaryMsg = new HashMap<>();
            summaryMsg.put("role", ConversationSummaryService.SUMMARY_ROLE);
            summaryMsg.put("content", summaryContent);
            messageHistory.add(0, summaryMsg);
        }

        private static Map<String, String> messageOf(String role, String content) {
            Map<String, String> msg = new HashMap<>();
            msg.put("role", role);
            msg.put("content", content);
            return msg;
        }
    }
}
