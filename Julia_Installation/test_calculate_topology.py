from pyxtal.db import database_topology


if __name__ == "__main__":
    db1 = database_topology("debug.db")
     
    db1.update_row_topology(prefix="mof-0-env", timeout=60)
