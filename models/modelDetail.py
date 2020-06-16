from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, PickleType
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AiModelDetail(Base):
    '''This is ReceiveJobs sample Data model class.'''
    
    __tablename__ = "tModelDetails"
    __table_args__ = {"schema":"KnowHow.dbo"}

    id = Column(Integer, primary_key=True, nullable=False)
    port = Column(Text, nullable=False)
    url = Column(Text, nullable=True)
    version = Column(Text, nullable=True)
    modelJson = Column(Integer, nullable=True)
    status = Column(DateTime, nullable=True)
    modelID = Column(Integer, nullable=True)
    createdBy = Column(Integer, nullable=True)
    createdOn = Column(DateTime, nullable=True)
    modifiedBy = Column(Integer, nullable=True)
    modifiedOn = Column(DateTime, nullable=True)
        
    def __repr__(self):
        return '<ReceiveJobs model {}>'.format(self.id)