from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, PickleType
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AiModel(Base):
    '''This is AiModel sample Data model class.'''
    
    __tablename__ = "tModels"
    __table_args__ = {"schema":"KnowHow.dbo"}

    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(Text, nullable=False)
    modelName = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    createdBy = Column(Integer, nullable=True)
    createdOn = Column(DateTime, nullable=True)
    modifiedBy = Column(Integer, nullable=True)
    modifiedOn = Column(DateTime, nullable=True)
    isActive = Column(Boolean, nullable=True)
    categoryId = Column(Integer, nullable=True)    
        
    def __repr__(self):
        return '<AiModel model {}>'.format(self.id)