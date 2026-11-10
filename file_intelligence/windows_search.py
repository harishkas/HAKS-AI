"""Windows Search Integration for HAKS File Intelligence (Phase 9.2).

Interfaces with Windows Search Indexer via OLE DB (Provider=Search.CollatorDSO)
for fast OS-level filesystem metadata and indexed document content search.
"""

import logging
from pathlib import Path
from typing import Any, Optional

try:
    import win32com.client
except ImportError:
    win32com = None

from file_intelligence.models import SearchResultItem

logger = logging.getLogger("haks.file_intelligence.windows_search")


class WindowsSearchEngine:
    """Queries native Windows Search SystemIndex for fast metadata and content lookup."""

    def __init__(self) -> None:
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check whether Windows Search OLE DB provider is accessible."""
        if self._available is not None:
            return self._available

        if win32com is None:
            self._available = False
            return False

        try:
            conn = win32com.client.Dispatch("ADODB.Connection")
            conn.Open("Provider=Search.CollatorDSO;Extended Properties='Application=Windows';")
            conn.Close()
            self._available = True
        except Exception as exc:
            logger.debug("Windows Search provider not available: %s", exc)
            self._available = False

        return self._available

    def search_by_name(
        self,
        query: str,
        directory_scope: Optional[str] = None,
        max_results: int = 20,
    ) -> list[SearchResultItem]:
        """Search indexed files matching filename or extension."""
        if not self.is_available():
            return []

        clean_term = query.replace("'", "''").strip()
        if not clean_term:
            return []

        # Build SQL query for Windows Search SystemIndex
        scope_clause = ""
        if directory_scope and Path(directory_scope).exists():
            scope_dir = str(Path(directory_scope).resolve()).replace("'", "''")
            scope_clause = f" AND SCOPE = 'file:{scope_dir}'"

        sql = (
            f"SELECT TOP {max_results} "
            "System.ItemPathDisplay, System.ItemNameDisplay, System.ItemType, System.Size, System.DateModified "
            f"FROM SystemIndex "
            f"WHERE System.ItemNameDisplay LIKE '%{clean_term}%' {scope_clause} "
            "ORDER BY System.DateModified DESC"
        )

        return self._execute_sql(sql)

    def search_by_content(
        self,
        keywords: str,
        directory_scope: Optional[str] = None,
        max_results: int = 20,
    ) -> list[SearchResultItem]:
        """Search indexed document content using FREETEXT or CONTAINS."""
        if not self.is_available():
            return []

        clean_kw = keywords.replace("'", "''").strip()
        if not clean_kw:
            return []

        scope_clause = ""
        if directory_scope and Path(directory_scope).exists():
            scope_dir = str(Path(directory_scope).resolve()).replace("'", "''")
            scope_clause = f" AND SCOPE = 'file:{scope_dir}'"

        sql = (
            f"SELECT TOP {max_results} "
            "System.ItemPathDisplay, System.ItemNameDisplay, System.ItemType, System.Size, System.DateModified, System.Search.Rank "
            f"FROM SystemIndex "
            f"WHERE CONTAINS(System.Search.Contents, '\"{clean_kw}\"') {scope_clause} "
            "ORDER BY System.Search.Rank DESC"
        )

        return self._execute_sql(sql)

    def _execute_sql(self, sql: str) -> list[SearchResultItem]:
        """Execute ADODB query against Windows Search provider."""
        results: list[SearchResultItem] = []
        try:
            conn = win32com.client.Dispatch("ADODB.Connection")
            conn.Open("Provider=Search.CollatorDSO;Extended Properties='Application=Windows';")
            rs = win32com.client.Dispatch("ADODB.Recordset")
            rs.Open(sql, conn)

            while not rs.EOF:
                path_val = rs.Fields("System.ItemPathDisplay").Value
                name_val = rs.Fields("System.ItemNameDisplay").Value
                ext_val = Path(path_val).suffix.lower() if path_val else ""
                size_val = rs.Fields("System.Size").Value or 0

                if path_val and Path(path_val).exists():
                    results.append(
                        SearchResultItem(
                            file_path=str(path_val),
                            file_name=str(name_val or Path(path_val).name),
                            file_type=ext_val,
                            snippet=f"File: {name_val} ({size_val // 1024} KB)",
                            score=0.85,
                            match_source="windows_search",
                            metadata={"size": size_val},
                        )
                    )
                rs.MoveNext()

            rs.Close()
            conn.Close()
        except Exception as exc:
            logger.debug("Windows Search execution notice for SQL [%s]: %s", sql, exc)

        return results

