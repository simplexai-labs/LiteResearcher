#!/usr/bin/env python3
"""
PostgreSQL 全文取数（检索后端用）

向量检索只返回截断的 snippet；当导入时启用了 ENABLE_SQL_STORAGE，全文会写入
PostgreSQL。本模块提供"按 URL 取整篇原文"的轻量查询，支撑 /web_parser、
/get_document 接口。

刻意不依赖 rag_core（避免拖入 torch/transformers）；只用 psycopg2 连接池。
表结构（由导入端创建）：{SQL_SCHEMA}.{SQL_TABLE}(url UNIQUE, title, text, ...)
"""

from typing import Dict, List, Optional

from diskann_config import (
    ENABLE_SQL_FULLTEXT, SQL_HOST, SQL_PORT, SQL_DATABASE, SQL_USER, SQL_PASSWORD,
    SQL_SCHEMA, SQL_TABLE, SQL_POOL_MIN, SQL_POOL_MAX,
)

_pool = None


def init_pool() -> bool:
    """初始化连接池。SQL 未启用或 psycopg2 缺失时返回 False（后端可正常退化为只做向量检索）。"""
    global _pool
    if not ENABLE_SQL_FULLTEXT:
        print("ℹ️  SQL 全文未启用（ENABLE_SQL_FULLTEXT=false），/web_parser 将不可用")
        return False
    try:
        from psycopg2.pool import ThreadedConnectionPool
        from psycopg2.extras import RealDictCursor  # noqa: F401  (查询时使用)
    except ImportError:
        print("⚠️  未安装 psycopg2，无法启用 SQL 全文取数。pip install psycopg2-binary")
        return False

    try:
        _pool = ThreadedConnectionPool(
            SQL_POOL_MIN, SQL_POOL_MAX,
            host=SQL_HOST, port=SQL_PORT, dbname=SQL_DATABASE,
            user=SQL_USER, password=SQL_PASSWORD,
        )
        # 连通性探测
        conn = _pool.getconn()
        _pool.putconn(conn)
        print(f"✅ PostgreSQL 连接池就绪: {SQL_HOST}:{SQL_PORT}/{SQL_DATABASE} "
              f"表 {SQL_SCHEMA}.{SQL_TABLE}")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL 连接池初始化失败: {e}")
        _pool = None
        return False


def close_pool():
    """关闭连接池。"""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def is_ready() -> bool:
    return _pool is not None


def get_fulltext_by_url(url: str) -> Optional[Dict]:
    """按 URL 取整篇原文，返回 {url, title, text}；不存在或未启用时返回 None。"""
    if _pool is None:
        return None
    from psycopg2.extras import RealDictCursor
    conn = None
    try:
        conn = _pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT url, title, text FROM {SQL_SCHEMA}.{SQL_TABLE} WHERE url = %s LIMIT 1",
                (url,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as e:
        print(f"❌ 按 URL 取全文失败: {e}")
        return None
    finally:
        if conn is not None:
            _pool.putconn(conn)


def get_fulltext_by_urls(urls: List[str]) -> Dict[str, Dict]:
    """批量按 URL 取全文，返回 {url: {url,title,text}}（只含命中的）。"""
    if _pool is None or not urls:
        return {}
    from psycopg2.extras import RealDictCursor
    conn = None
    try:
        conn = _pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT url, title, text FROM {SQL_SCHEMA}.{SQL_TABLE} WHERE url = ANY(%s)",
                (list(urls),),
            )
            return {r["url"]: dict(r) for r in cur.fetchall()}
    except Exception as e:
        print(f"❌ 批量取全文失败: {e}")
        return {}
    finally:
        if conn is not None:
            _pool.putconn(conn)
