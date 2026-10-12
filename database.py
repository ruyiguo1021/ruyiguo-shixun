import pymysql


class DatabaseManager:

  def __init__(
      self, host="localhost", user="root", password="", database="helmet_db"
  ):
    self.host = host
    self.user = user
    self.password = password
    self.database = database
    self.init_database()

  def get_connection(self):
    return pymysql.connect(
        host=self.host,
        user=self.user,
        password=self.password,
        database=self.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

  def init_database(self):
    """初始化数据库及相关表"""
    # 1. 确保数据库存在
    conn = pymysql.connect(
        host=self.host, user=self.user, password=self.password, charset="utf8mb4"
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database};")
    conn.close()

    # 2. 创建告警日志表与用户权限表
    conn = self.get_connection()
    cursor = conn.cursor()

    # 告警日志表
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS alarm_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                wei_dai_count INT NOT NULL,
                yi_dai_count INT NOT NULL,
                snapshot_path VARCHAR(255) NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

    # 用户表（用于 RBAC 权限控制）
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(50) NOT NULL,
                role VARCHAR(20) NOT NULL -- 'admin' 或 'operator'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

    # 预置默认账号（如果表为空）
    cursor.execute("SELECT COUNT(*) AS cnt FROM users;")
    res = cursor.fetchone()
    if res["cnt"] == 0:
      default_users = [
          ("admin", "123456", "admin"),
          ("user", "123456", "operator"),
      ]
      cursor.executemany(
          "INSERT INTO users (username, password, role) VALUES (%s, %s, %s);",
          default_users,
      )
      conn.commit()

    cursor.close()
    conn.close()

  def verify_user(self, username, password):
    """从数据库校验用户登录，返回用户角色（admin 或 operator），失败返回 None"""
    conn = self.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role FROM users WHERE username = %s AND password = %s;",
        (username, password),
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
      return result["role"]
    return None

  def add_alarm_record(self, wei_dai_count, yi_dai_count, snapshot_path):
    conn = self.get_connection()
    cursor = conn.cursor()
    sql = "INSERT INTO alarm_logs (wei_dai_count, yi_dai_count, snapshot_path) VALUES (%s, %s, %s)"
    cursor.execute(sql, (wei_dai_count, yi_dai_count, snapshot_path))
    conn.commit()
    cursor.close()
    conn.close()

  def get_alarm_records(self):
    conn = self.get_connection()
    cursor = conn.cursor()
    sql = "SELECT id, wei_dai_count, yi_dai_count, snapshot_path, timestamp FROM alarm_logs ORDER BY timestamp DESC"
    cursor.execute(sql)
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records

  def delete_alarm_record(self, record_id):
    """根据日志 ID 删除数据库中的违规记录"""
    conn = self.get_connection()
    cursor = conn.cursor()
    try:
      sql = "DELETE FROM alarm_logs WHERE id = %s"
      cursor.execute(sql, (record_id,))
      conn.commit()
    except Exception as e:
      print(f"删除记录失败: {e}")
    finally:
      cursor.close()
      conn.close()