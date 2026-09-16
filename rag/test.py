from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", 'VeSBEeylDgvlqppqv2sJpZX9bBxe8IMTO6i7QP0haso'))
with driver.session() as session:
    session.run("CREATE (n:Test {name: 'hello'})")
    print(session.run("MATCH (n:Test) RETURN n").data())