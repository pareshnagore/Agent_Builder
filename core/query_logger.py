"""
Query Logger for Agent_ng
Logs and analyzes queries for observability and debugging.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List


class QueryLoggerException(Exception):
    """Base exception for query logger errors."""
    pass


class QueryLogger:
    """
    Logs queries and retrieval results for analysis and debugging.
    
    Usage:
        logger = QueryLogger(logs_dir="data/logs")
        log_id = logger.log({"query": "test", "results": []})
        logs = logger.read_logs(limit=100)
    """

    def __init__(self, logs_dir: str = "data/logs"):
        """
        Initialize query logger.
        
        Args:
            logs_dir: Directory to store log files
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        self.log_file = self.logs_dir / "queries.jsonl"

    def log(self, entry: dict) -> str:
        """
        Log a single entry.
        
        Args:
            entry: Entry to log (dict)
        
        Returns:
            Entry ID (timestamp-based)
        """
        # Ensure timestamp
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        # Generate entry ID
        entry_id = entry.get("id", self._generate_id())
        entry["id"] = entry_id

        try:
            # Append to JSONL file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            return entry_id

        except Exception as e:
            raise QueryLoggerException(f"Failed to log entry: {e}")

    def read_logs(self, limit: int = 100, filters: Optional[dict] = None) -> List[dict]:
        """
        Read recent log entries.
        
        Args:
            limit: Max entries to return
            filters: Optional filters (e.g., {"source": "document.pdf"})
        
        Returns:
            List of entries
        """
        if not self.log_file.exists():
            return []

        try:
            entries = []
            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)

                        # Apply filters
                        if filters:
                            if not self._matches_filters(entry, filters):
                                continue

                        entries.append(entry)

            # Return most recent entries first
            return entries[-limit:][::-1]

        except Exception as e:
            raise QueryLoggerException(f"Failed to read logs: {e}")

    def search_logs(self, query_text: str) -> List[dict]:
        """
        Search logs by query text.
        
        Args:
            query_text: Text to search for (case-insensitive)
        
        Returns:
            List of matching entries
        """
        query_text = query_text.lower()
        results = []

        try:
            if not self.log_file.exists():
                return []

            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if (
                            "query" in entry
                            and query_text in entry["query"].lower()
                        ):
                            results.append(entry)

            return results

        except Exception as e:
            raise QueryLoggerException(f"Failed to search logs: {e}")

    def get_stats(self) -> dict:
        """
        Get statistics from logs.
        
        Returns:
            Dict with stats
        """
        try:
            logs = self.read_logs(limit=10000)

            if not logs:
                return {
                    "total_queries": 0,
                    "avg_results_per_query": 0,
                    "top_sources": [],
                }

            total_queries = len(logs)
            avg_results = (
                sum(entry.get("num_results", 0) for entry in logs) / total_queries
                if total_queries > 0
                else 0
            )

            # Get top sources
            sources = {}
            for entry in logs:
                for source in entry.get("sources", []):
                    if source:
                        sources[source] = sources.get(source, 0) + 1

            top_sources = sorted(
                sources.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return {
                "total_queries": total_queries,
                "avg_results_per_query": round(avg_results, 2),
                "top_sources": [{"source": s, "count": c} for s, c in top_sources],
                "log_file": str(self.log_file),
            }

        except Exception as e:
            raise QueryLoggerException(f"Failed to get stats: {e}")

    def export_logs(self, output_path: str = "query_logs_export.json") -> str:
        """
        Export all logs to JSON.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to exported file
        """
        try:
            logs = self.read_logs(limit=10000)
            with open(output_path, "w") as f:
                json.dump(logs, f, indent=2)

            return output_path

        except Exception as e:
            raise QueryLoggerException(f"Failed to export logs: {e}")

    def clear_old_logs(self, days: int = 30) -> int:
        """
        Clear logs older than specified days.
        
        Args:
            days: Age threshold in days
        
        Returns:
            Number of entries removed
        """
        try:
            from datetime import datetime, timedelta

            if not self.log_file.exists():
                return 0

            cutoff = datetime.now() - timedelta(days=days)
            remaining = []
            removed = 0

            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        timestamp_str = entry.get("timestamp")

                        if timestamp_str:
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str)
                                if timestamp >= cutoff:
                                    remaining.append(entry)
                                else:
                                    removed += 1
                            except ValueError:
                                remaining.append(entry)
                        else:
                            remaining.append(entry)

            # Rewrite log file
            with open(self.log_file, "w") as f:
                for entry in remaining:
                    f.write(json.dumps(entry) + "\n")

            return removed

        except Exception as e:
            raise QueryLoggerException(f"Failed to clear old logs: {e}")

    # ========== PRIVATE METHODS ==========

    @staticmethod
    def _generate_id() -> str:
        """Generate unique ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

    @staticmethod
    def _matches_filters(entry: dict, filters: dict) -> bool:
        """Check if entry matches all filters."""
        for key, value in filters.items():
            if key == "sources":
                # Special handling for sources list
                if value not in entry.get("sources", []):
                    return False
            else:
                if entry.get(key) != value:
                    return False

        return True
