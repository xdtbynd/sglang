"""
Cache adapter that wraps the suffix decoding backend cache to provide
the same interface as NgramCache.

This allows NGRAMWorker to use suffix decoding without modification.
"""

import logging
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SuffixCacheAdapter:
    """
    Adapter that wraps SuffixDecodingCache to match NgramCache interface.

    NGRAMWorker expects:
    - batch_get(batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]
      Returns (draft_tokens, tree_mask) as flat numpy arrays
    - batch_put(batch_tokens: List[List[int]]) -> None
      Updates cache with verified tokens
    - synchronize() -> None
      No-op for suffix cache
    - reset() -> None
      Clears all cached data
    """

    def __init__(
        self,
        draft_token_num: int,
        max_batch_size: int,
        max_tree_depth: int = 24,
        max_cached_requests: int = 10000,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        """
        Args:
            draft_token_num: Fixed number of draft tokens (for padding)
            max_tree_depth: Maximum depth for suffix tree
            max_cached_requests: Maximum number of cached requests
            max_spec_factor: Maximum speculation factor
            min_token_prob: Minimum token probability threshold
        """
        # Lazy import to avoid error when Suffix Decoding is not used
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.draft_token_num = draft_token_num
        self.max_batch_size = max_batch_size
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob

        # Debug toggles (set env e.g. SUFFIX_DEBUG_TREE=1 to dump first batch)
        self.debug_tree_dump_remaining = int(os.environ.get("SUFFIX_DEBUG_TREE", "0"))

        # Track state by SGlang request ID (stable identifier)
        # Map: sglang_req_id → (arctic_req_id, last_length)
        self.req_state = {}

        # Preallocate buffers to avoid per-step allocations
        self.max_total_drafts = self.max_batch_size * self.draft_token_num
        self.draft_buffer = np.empty((self.max_total_drafts,), dtype=np.int64)
        self.mask_buffer = np.empty(
            (self.max_batch_size, self.draft_token_num, self.draft_token_num),
            dtype=bool,
        )

    def _cleanup_inactive_requests(self, active_req_ids: set[str]):
        """Stop backend requests that are no longer active in SGlang."""
        inactive_req_ids = [
            rid for rid in self.req_state.keys() if rid not in active_req_ids
        ]
        for rid in inactive_req_ids:
            cache_req_id, _ = self.req_state.pop(rid)
            if cache_req_id in getattr(self.suffix_cache, "active_requests", set()):
                self.suffix_cache.stop_request(cache_req_id)

    def _get_or_create_cache_req_id(
        self, sglang_req_id: str, prompt: List[int], tokens: List[int]
    ) -> tuple:
        """Get or create a backend request ID for the given SGlang request.

        Args:
            sglang_req_id: Stable request ID from SGlang
            prompt: Prompt tokens only (no generated tokens)
            tokens: Full token sequence (prompt + outputs)

        Returns: (arctic_req_id, last_length)
        """
        if sglang_req_id not in self.req_state:
            # Use SGlang request ID directly as backend request ID
            cache_req_id = sglang_req_id

            # Initialize the request in suffix cache with ONLY the prompt
            self.suffix_cache.start_request(cache_req_id, prompt)

            # Track: [arctic_req_id, last_length]
            # IMPORTANT: Set last_length to prompt length since the backend already has the prompt
            self.req_state[sglang_req_id] = [cache_req_id, len(prompt)]

        cache_req_id, last_length = self.req_state[sglang_req_id]
        return cache_req_id, last_length

    def batch_get(
        self,
        batch_req_ids: List[str],
        batch_prompts: List[List[int]],
        batch_tokens: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get draft tokens for a batch of token sequences.

        This is called BEFORE verification with the current state.
        We speculate based on the current tokens.

        Args:
            batch_req_ids: List of SGlang request IDs (stable)
            batch_prompts: List of prompt tokens (no generated tokens)
            batch_tokens: List of token sequences (prompt + output tokens)

        Returns:
            Tuple of:
            - draft_tokens: np.ndarray of shape (batch_size * draft_token_num,)
            - tree_mask: np.ndarray of shape (batch_size * draft_token_num * draft_token_num,)
        """
        batch_size = len(batch_req_ids)
        if batch_size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds configured max_batch_size={self.max_batch_size}"
            )

        total_draft_tokens = batch_size * self.draft_token_num
        draft_view = self.draft_buffer[:total_draft_tokens]
        mask_view = self.mask_buffer[:batch_size]
        mask_view.fill(False)

        active_req_ids = set(batch_req_ids)
        self._cleanup_inactive_requests(active_req_ids)

        for idx, (sglang_req_id, prompt, tokens) in enumerate(
            zip(batch_req_ids, batch_prompts, batch_tokens)
        ):
            cache_req_id, last_length = self._get_or_create_cache_req_id(
                sglang_req_id, prompt, tokens
            )

            # Ensure cache includes the latest verified tokens before speculation.
            current_length = len(tokens)
            if current_length > last_length:
                new_tokens = tokens[last_length:current_length]
                if cache_req_id in self.suffix_cache.active_requests:
                    self.suffix_cache.add_active_response(cache_req_id, new_tokens)
                    self.req_state[sglang_req_id][1] = current_length
                    last_length = current_length
                else:
                    logger.warning(
                        f"[BATCH_GET {idx}] Suffix cache req {cache_req_id} not active when updating!"
                    )

            # Extract pattern from end of tokens (up to max_tree_depth)
            pattern_start = max(0, len(tokens) - self.max_tree_depth)
            pattern = tokens[pattern_start:]

            # Speculate using suffix cache
            draft = self.suffix_cache.speculate(
                cache_req_id,
                pattern,
                max_spec_tokens=self.draft_token_num,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            # Convert to fixed-size arrays
            draft_ids = list(draft.token_ids)
            draft_parents = list(draft.parents)
            draft_ids, draft_parents = self._reorder_tree_bfs(draft_ids, draft_parents)
            
            # Apply language model level constraints to reduce repetition
            draft_ids, draft_parents = self._filter_repetitive_sequences(draft_ids, draft_parents, tokens)

            context_token = tokens[-1] if tokens else 0
            draft_ids, draft_parents = self._inject_root_node(
                draft_ids, draft_parents, context_token
            )

            # Pad or truncate to match draft_token_num (includes root node at index 0)
            original_draft_len = len(draft_ids)
            if original_draft_len < self.draft_token_num:
                pad_len = self.draft_token_num - original_draft_len
                draft_ids.extend([0] * pad_len)
                draft_parents.extend([0] * pad_len)
            elif original_draft_len > self.draft_token_num:
                draft_ids = draft_ids[: self.draft_token_num]
                draft_parents = draft_parents[: self.draft_token_num]
                original_draft_len = self.draft_token_num

            start = idx * self.draft_token_num
            end = start + self.draft_token_num
            draft_view[start:end] = draft_ids

            # Build tree mask from parent structure
            # Token i can attend to token j if j is an ancestor of i
            mask = mask_view[idx]
            if original_draft_len > 0:
                for i in range(original_draft_len):
                    mask[i, i] = True  # Self-attention
                    parent_idx = draft_parents[i]
                    while parent_idx >= 0 and parent_idx < self.draft_token_num:
                        mask[i, parent_idx] = True
                        parent_idx = draft_parents[parent_idx]

            if self.debug_tree_dump_remaining > 0 and original_draft_len > 0:
                logger.warning(
                    "[SUFFIX DEBUG] req=%s, original_draft_len=%d, masked_len=%d, draft_ids=%s",
                    sglang_req_id,
                    original_draft_len,
                    len(draft_ids),
                    draft_ids,
                )
                logger.warning(
                    "[SUFFIX DEBUG] mask=\n%s",
                    mask.astype(int),
                )
                self.debug_tree_dump_remaining -= 1

        tree_mask = mask_view.reshape(-1)[: total_draft_tokens * self.draft_token_num]

        return draft_view, tree_mask

    def batch_put(self, batch_req_ids: List[str], batch_tokens: List[List[int]]):
        """
        No-op: cache updates now happen inside batch_get before speculation.
        Kept for interface compatibility with NGRAMWorker.
        """
        _ = batch_tokens  # Intentional placeholder to satisfy interface.
        for idx, sglang_req_id in enumerate(batch_req_ids):
            if sglang_req_id not in self.req_state:
                logger.error(
                    f"[BATCH_PUT {idx}] Called for unknown request {sglang_req_id}! "
                    f"This should not happen - batch_get must be called first."
                )

    def synchronize(self):
        """No-op for suffix cache (no async operations)."""
        pass

    def reset(self):
        """Clear all cached data."""
        # Stop all active requests
        for cache_req_id in list(self.suffix_cache.active_requests):
            self.suffix_cache.stop_request(cache_req_id)
        # Clear tracking
        self.req_state.clear()
        logger.info("[SUFFIX ADAPTER] Cache reset")

    def _reorder_tree_bfs(
        self, token_ids: List[int], parents: List[Optional[int]]
    ) -> Tuple[List[int], List[int]]:
        """
        Reorder nodes so parents always precede their descendants.

        reconstruct_indices_from_tree_mask assumes this layout; the backend emits
        score-ordered nodes, so we re-topologize the list before building masks.
        """
        n = len(token_ids)
        if n <= 1:
            return token_ids, parents

        children: List[List[int]] = [[] for _ in range(n)]
        roots: List[int] = []
        for idx, parent in enumerate(parents):
            if parent is None or parent < 0 or parent >= n:
                roots.append(idx)
            else:
                children[parent].append(idx)

        if not roots:
            roots = [0]

        order: List[int] = []
        visited = [False] * n
        for root in roots:
            if visited[root]:
                continue
            queue = deque([root])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                order.append(node)
                for child in children[node]:
                    if not visited[child]:
                        queue.append(child)

        # Append any detached nodes (should not happen, but keep deterministic order).
        for idx in range(n):
            if not visited[idx]:
                order.append(idx)

        if order == list(range(n)):
            return token_ids, parents

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
        reordered_ids = [token_ids[old_idx] for old_idx in order]
        reordered_parents: List[int] = []
        for old_idx in order:
            parent = parents[old_idx]
            if parent is None or parent < 0:
                reordered_parents.append(-1)
            else:
                reordered_parents.append(remap.get(parent, -1))

        return reordered_ids, reordered_parents

    def _filter_repetitive_sequences(
        self, token_ids: List[int], parents: List[Optional[int]], context_tokens: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply language model level constraints to reduce repetition in draft tokens.
        This method implements several mechanisms to ensure diversity in generated content:
        
        1. n-gram repetition restriction: Prevents repeating n-grams (3-5 tokens) from context
        2. Sequence similarity detection: Avoids generating sequences too similar to recent content
        3. Token frequency balancing: Ensures a balanced distribution of tokens
        4. Pattern repetition detection: Identifies and avoids repetitive patterns
        
        Args:
            token_ids: List of draft token IDs
            parents: List of parent indices
            context_tokens: Current context token sequence
            
        Returns:
            Filtered token_ids and parents lists with preserved tree structure
        """
        if not token_ids:
            return token_ids, parents
            
        # We don't remove nodes to preserve tree structure, instead we replace 
        # repetitive tokens with 0 (padding) while keeping the parent structure intact
        filtered_ids = []
        filtered_parents = []
        
        # Get recent context for n-gram checking
        recent_context = context_tokens[-50:]  # Larger context window for better repetition detection
        
        # Extract all n-grams from recent context (n=3,4,5)
        context_ngrams = set()
        for n in [3, 4, 5]:
            for i in range(len(recent_context) - n + 1):
                ngram = tuple(recent_context[i:i+n])
                context_ngrams.add(ngram)
        
        # Track generated tokens in the current draft
        current_draft = []
        
        # Track token frequency to ensure balance
        token_freq = {}
        for token in recent_context:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        # Maximum allowed consecutive duplicates
        MAX_CONSECUTIVE_DUP = 2
        consecutive_counts = {}
        
        for i, (token, parent) in enumerate(zip(token_ids, parents)):
            # Reset consecutive count for other tokens
            for t in consecutive_counts:
                if t != token:
                    consecutive_counts[t] = 0
            consecutive_counts[token] = consecutive_counts.get(token, 0) + 1
            
            is_excessive_repetition = False
            
            # 1. Check consecutive duplicates
            if consecutive_counts[token] > MAX_CONSECUTIVE_DUP:
                is_excessive_repetition = True
            
            # 2. Check if token would create a repeated n-gram with current draft
            check_seq = current_draft[-4:] + [token]  # Check for 3,4,5-grams
            for n in [3, 4, 5]:
                if len(check_seq) >= n:
                    ngram = tuple(check_seq[-n:])
                    if ngram in context_ngrams:
                        is_excessive_repetition = True
                        break
            
            # 3. Check token frequency balance
            current_freq = token_freq.get(token, 0) + 1  # +1 for the current token
            if current_freq > 5:  # Allow maximum 5 occurrences of any token in context
                is_excessive_repetition = True
            
            # 4. Check for repeated patterns (e.g., A-B-A-B, A-B-C-A-B-C)
            if len(current_draft) >= 3:
                # Check 2-token pattern repetition
                if len(current_draft) >= 4:
                    if (current_draft[-1] == current_draft[-3] and 
                        token == current_draft[-2]):
                        is_excessive_repetition = True
                # Check 3-token pattern repetition
                if len(current_draft) >= 6:
                    if (current_draft[-2] == current_draft[-5] and 
                        current_draft[-1] == current_draft[-4] and 
                        token == current_draft[-3]):
                        is_excessive_repetition = True
            
            if not is_excessive_repetition:
                filtered_ids.append(token)
                current_draft.append(token)
                # Update token frequency
                token_freq[token] = token_freq.get(token, 0) + 1
            else:
                # Replace with 0 (padding token) to reduce repetition but keep structure
                filtered_ids.append(0)
            
            # Always keep the parent structure intact
            filtered_parents.append(parent)
        
        # If all tokens were filtered, keep at least the first one
        if not any(t != 0 for t in filtered_ids):
            filtered_ids[0] = token_ids[0]
            
        return filtered_ids, filtered_parents

    def _inject_root_node(
        self, token_ids: List[int], parents: List[int], context_token: int
    ) -> Tuple[List[int], List[int]]:
        """
        Insert the latest verified token as index 0 so the layout matches NGRAM.
        """
        rooted_ids = [context_token]
        rooted_parents = [-1]
        for parent_idx in parents:
            if parent_idx < 0:
                rooted_parents.append(0)
            else:
                rooted_parents.append(parent_idx + 1)
        rooted_ids.extend(token_ids)
        return rooted_ids, rooted_parents
