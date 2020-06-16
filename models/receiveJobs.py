from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, PickleType, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ReceiveJobs(Base):
    '''This is ReceiveJobs sample Data model class.'''
    
    __tablename__ = "tReceiveJobs"
    __table_args__ = {"schema":"KnowHow.dbo"}

    id = Column(Integer, primary_key=True, nullable=False)
    responseURL = Column(Text, nullable=False)
    requestExecutionTime = Column(Text, nullable=True)
    requestStatus = Column(Text, nullable=True)
    keyType = Column(Text, nullable=True)
    projectID = Column(Integer, nullable=True)
    unProcessedImage = Column(Text, nullable=True)
    unProcessedPath = Column(Text, nullable=True)
    processedImage = Column(Text, nullable=True)
    processedPath = Column(Text, nullable=True)
    uploadType = Column(Text, nullable=True)
    inputPublicId = Column(Text, nullable=True)
    outputPublicId = Column(Text, nullable=True)
    fullyQualifiedPublicId = Column(Text, nullable=True)
    uri = Column(Text, nullable=True)
    localPath = Column(Text, nullable=True)
    createdOn = Column(DateTime, nullable=True)
    modifiedOn = Column(DateTime, nullable=True)
        
    def __repr__(self):
        return '<ReceiveJobs model {}>'.format(self.id)