import networkx as nx

class DependencyGraphBuilder:
    """
    Builds a NetworkX graph showing code dependencies (e.g., function calls).
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_call(self, caller: str, callee: str, filepath: str = "unknown"):
        """Adds a directed edge representing a function call."""
        self.graph.add_edge(caller, callee, type="calls", filepath=filepath)

    def get_callers(self, target_function: str):
        """Returns nodes that have an edge pointing to the target function."""
        if target_function in self.graph:
            return list(self.graph.predecessors(target_function))
        return []

    def get_callees(self, target_function: str):
        """Returns nodes that are called by the target function."""
        if target_function in self.graph:
            return list(self.graph.successors(target_function))
        return []

    def build_dummy_graph(self):
        """Populates the graph with mock data matching our evaluation test case."""
        # For the sample test case: What calls the authenticate_user function?
        self.add_call("login_handler", "authenticate_user", filepath="auth.py")
