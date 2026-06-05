import psycopg2
from psycopg2 import sql

def check_table_content():
    try:
        # 连接数据库
        conn = psycopg2.connect(
            host="47.111.147.142",
            port="8432",
            database="postgres",
            user="postgres",
            password="pass123"
        )
        cur = conn.cursor()

        # 1. 首先查看表结构 (Column names and types)
        print("--- Table Structure for 'serper_wiki' ---")
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'serper_wiki'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        for col in columns:
            print(f"Column: {col[0]:<20} | Type: {col[1]}")

        # 2. 查看表中的数据样例 (前 5 行)
        print("\n--- Data Samples (First 5 rows) ---")
        cur.execute("SELECT * FROM serper_wiki LIMIT 5;")
        
        # 获取列名用于显示
        colnames = [desc[0] for desc in cur.description]
        print(colnames)
        
        rows = cur.fetchall()
        for row in rows:
            print(row)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_table_content()