from langchain_community.tools import DuckDuckGoSearchRun, ShellTool #depricated

# https://docs.langchain.com/oss/python/integrations/tools#media-generation

search_tool = DuckDuckGoSearchRun()
shell_tool = ShellTool()

# results = search_tool.invoke('CJP Protest Update')
results = shell_tool.invoke('dir')

print(results)