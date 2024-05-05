import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create an SQLAlchemy engine to connect to the MySQL database
# Replace 'username', 'password', 'hostname', 'port', and 'database_name' with your MySQL configuration
engine = create_engine('mysql://username:password@hostname:port/database_name')

# Create a base class for declarative class definitions
Base = declarative_base()

# Define the Module class representing the 'module' table
class Module(Base):
    __tablename__ = 'module'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    parcours = Column(String(255))
    semestre = Column(String(255))

# Create the table in the database
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()



# Close the session
session.close()
