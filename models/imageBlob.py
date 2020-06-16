from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, PickleType
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ImageBlob(Base):
    '''This is ImageBlob sample Data model class.'''
    
    __tablename__ = "tImageBlob"
    __table_args__ = {"schema":"KnowHow.dbo"}

    id = Column(Integer, primary_key=True, nullable=False)
    apiKey = Column(Text, nullable=False)
    apiSecret = Column(Text, nullable=True)
    cloudName = Column(Text, nullable=True)
    destinationType = Column(Integer, nullable=True)
    isActive = Column(Boolean, nullable=True)
           
    def __repr__(self):
        return '<ImageBlob model {}>'.format(self.id)