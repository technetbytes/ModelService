from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, PickleType, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ProjectKey(Base):
    '''This is Project Keys sample Data model class.'''
    
    __tablename__ = "tProjectKeys"
    __table_args__ = {"schema":"KnowHow.dbo"}

    id = Column(Integer, primary_key=True, nullable=False)
    webKey = Column(Text, nullable=False)
    mobileKey = Column(Text, nullable=True)
    projectID = Column(Integer, nullable=True)
    createdBy = Column(Integer, nullable=True)
    createdOn = Column(DateTime, nullable=True)
    modifiedBy = Column(Integer, nullable=True)
    modifiedOn = Column(DateTime, nullable=True)
    isActive = Column(Boolean, nullable=True)
        
    def __repr__(self):
        return '<Project model {}>'.format(self.id)
