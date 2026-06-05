import psycopg2

try:
    conn = psycopg2.connect(
        host="47.111.147.142",
        port="8432",
        database="postgres",
        user="postgres",
        password="pass123"
    )
    cur = conn.cursor()
    # 查询 information_schema 获取表名
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cur.fetchall()
    for table in tables:
        print(table[0])
    cur.close()
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")