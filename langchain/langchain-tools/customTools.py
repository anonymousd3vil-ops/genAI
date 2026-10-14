from langchain.tools import tool

@tool
def addition(a: int, b: int) -> int:
    """This Adds given two number and returns the value"""
    return a+b

result = addition.invoke({'a': 10, 'b':20})

print(result)

print(addition.name)
print(addition.description)
print(addition.args)