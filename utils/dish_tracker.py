import sqlite3
import json
from typing import Dict, List, Optional
import os

class DishDevelopmentTracker:
    """
    菜品研发跟踪器，用于持久化保存菜品研发历史
    支持多轮迭代调整，记录每次修改的历史

    对应商家痛点：多轮打磨菜品，记不住之前的方案

    SQLite 持久化 vs LangGraph Checkpoint：
    - 本模块：存储结构化菜品数据（菜名、食材、文案、图片URL等）
    - LangGraph Checkpoint：存储对话状态（多轮对话上下文）
    - 两者互补，建议一起使用
    """

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(os.path.dirname(__file__), "..", "dish_development.db")
        self.db_path = os.path.abspath(db_path)
        self.init_db()

    def init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建菜品研发记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dishes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                dish_name TEXT NOT NULL,
                ingredients TEXT,
                recipe TEXT,
                cost_estimation TEXT,
                suggested_price TEXT,
                marketing_copy TEXT,
                image_url TEXT,
                dish_source TEXT DEFAULT 'ai',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                session_name TEXT DEFAULT '',
                restaurant_type TEXT DEFAULT '',
                location TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def save_session(self, session_id: str, metadata: Dict = None) -> bool:
        """保存或更新会话信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, session_name, restaurant_type, location)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
        ''', (
            session_id,
            (metadata or {}).get('session_name', ''),
            (metadata or {}).get('restaurant_type', ''),
            (metadata or {}).get('location', '')
        ))
        conn.commit()
        conn.close()
        return True

    def save_dish(self, session_id: str, dish_data: Dict) -> int:
        """
        保存一道菜品研发记录

        Args:
            session_id: 会话ID（Chainlit的session_id）
            dish_data: 菜品数据字典，格式：
                {
                    "dish_name": "秘制红烧鸡",
                    "ingredients": ["鸡肉500g", "土豆2个"],
                    "recipe": {"steps": ["...", "..."], "time": "30min"},
                    "cost_estimation": "8.5元/份",
                    "suggested_price": "19.9元",
                    "marketing_copy": {
                        "title": "...",
                        "description": "...",
                        "tags": ["..."]
                    },
                    "image_url": "https://oss...",
                    "dish_source": "ai"
                }

        Returns:
            菜品记录ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO dishes (
                session_id, dish_name, ingredients, recipe,
                cost_estimation, suggested_price, marketing_copy,
                image_url, dish_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            dish_data.get('dish_name', ''),
            json.dumps(dish_data.get('ingredients', []), ensure_ascii=False),
            json.dumps(dish_data.get('recipe', {}), ensure_ascii=False),
            dish_data.get('cost_estimation', ''),
            dish_data.get('suggested_price', ''),
            json.dumps(dish_data.get('marketing_copy', {}), ensure_ascii=False),
            dish_data.get('image_url', ''),
            dish_data.get('dish_source', 'ai')
        ))

        dish_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return dish_id

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """
        获取某次会话的所有菜品研发记录

        Args:
            session_id: 会话ID
            limit: 最多返回多少条

        Returns:
            菜品研发记录列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, session_id, dish_name, ingredients, recipe,
                   cost_estimation, suggested_price, marketing_copy,
                   image_url, dish_source, created_at
            FROM dishes
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (session_id, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def get_all_session_summaries(self) -> List[Dict]:
        """
        获取所有会话的概要信息（用于展示历史会话列表）

        Returns:
            每个会话的最新状态概要
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.session_id, s.session_name, s.restaurant_type,
                   s.location, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM dishes d WHERE d.session_id = s.session_id) as dish_count,
                   (SELECT d.dish_name FROM dishes d
                    WHERE d.session_id = s.session_id
                    ORDER BY d.created_at DESC LIMIT 1) as last_dish
            FROM sessions s
            ORDER BY s.updated_at DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        summaries = []
        for row in rows:
            summaries.append({
                'session_id': row[0],
                'session_name': row[1],
                'restaurant_type': row[2],
                'location': row[3],
                'created_at': row[4],
                'updated_at': row[5],
                'dish_count': row[6],
                'last_dish': row[7]
            })
        return summaries

    def find_dishes_by_ingredient(self, ingredient: str) -> List[Dict]:
        """
        根据食材关键词搜索历史菜品（跨会话搜索）

        Args:
            ingredient: 食材名称

        Returns:
            所有包含该食材的历史菜品
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, session_id, dish_name, ingredients, recipe,
                   cost_estimation, suggested_price, marketing_copy,
                   image_url, dish_source, created_at
            FROM dishes
            WHERE ingredients LIKE ?
            ORDER BY created_at DESC
            LIMIT 20
        ''', (f'%{ingredient}%',))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row) -> Dict:
        """将数据库行转换为字典"""
        return {
            'id': row[0],
            'session_id': row[1],
            'dish_name': row[2],
            'ingredients': json.loads(row[3]) if row[3] else [],
            'recipe': json.loads(row[4]) if row[4] else {},
            'cost_estimation': row[5],
            'suggested_price': row[6],
            'marketing_copy': json.loads(row[7]) if row[7] else {},
            'image_url': row[8],
            'dish_source': row[9],
            'created_at': row[10]
        }

# ---------- 全局单例，方便到处调用 ----------
_tracker_instance: Optional[DishDevelopmentTracker] = None

def get_tracker() -> DishDevelopmentTracker:
    """获取全局 DishDevelopmentTracker 实例"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = DishDevelopmentTracker()
    return _tracker_instance

def save_current_dish(session_id: str, dish_data: Dict) -> int:
    """快捷保存当前菜品"""
    return get_tracker().save_dish(session_id, dish_data)

def get_dish_history(session_id: str, limit: int = 20) -> List[Dict]:
    """快捷获取菜品研发历史"""
    return get_tracker().get_session_history(session_id, limit)

def get_all_user_sessions() -> List[Dict]:
    """快捷获取所有会话"""
    return get_tracker().get_all_session_summaries()

def search_dishes_by_ingredient(ingredient: str) -> List[Dict]:
    """快捷按食材搜索历史菜品"""
    return get_tracker().find_dishes_by_ingredient(ingredient)
