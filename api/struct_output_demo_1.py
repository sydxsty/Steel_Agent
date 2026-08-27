from my_llm import deepseek_Llm
from pydantic import BaseModel,Field

class movie(BaseModel):
    title:str = Field(description="电影标题")
    year:int = Field(description="电影上映年份")
    directer:str = Field(description="电影导演姓名")
    reting:float = Field(description="评分")
deepseek_Llm.with_structured_output(movie)
resp = deepseek_Llm.invoke("介绍一下电影，肖申克救赎")
print(type(resp))#如果有返回则返回结构类型movie，如果没有返回则返回AIMessage类型content
print(resp)