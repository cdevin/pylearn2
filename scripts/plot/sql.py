import psycopg2

class SQL():

    def __init__(self):

        self.conn = psycopg2.connect(host = "gershwin.iro.umontreal.ca", database = "mirzamom_db",
                user = "mirzamom", password = "pishy83")
        self.cur = self.conn.cursor()

    def __del__(self):

        self.cur.close()
        self.conn.close()

    def get_many(self, command, num):

        self.cur.execute(command)
        return self.cur.fetchmany(num)

    def get_one(self, command):

        self.cur.execute(command)
        return self.cur.fetchone()

    def get_all(self, command):

        self.cur.execute(command)
        return self.cur.fetchall()


