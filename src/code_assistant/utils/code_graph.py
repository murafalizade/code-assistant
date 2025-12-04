import json
import os

import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser

TS_LANGUAGE = tstypescript.language_typescript()

parser = Parser(Language(TS_LANGUAGE))


def read_file(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def get_text(code, node):
    return code[node.start_byte : node.end_byte]


def build_graph_for_file(path):
    code = read_file(path)
    print(code, "code")
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node

    nodes = {}
    edges = []

    file_id = path

    for node in walk(root):
        kind = node.type

        if kind == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = get_text(code, name_node)
                class_id = f"{file_id}:{class_name}"

                nodes[class_id] = {"type": "class", "name": class_name, "file": file_id}

        if kind == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                method_name = get_text(code, name_node)
                method_id = f"{file_id}:{method_name}"

                nodes[method_id] = {"type": "method", "name": method_name, "file": file_id}

        if kind == "decorator":
            expr = node.child_by_field_name("expression")
            if expr:
                decorator_name = get_text(code, expr)
                edges.append({"from": file_id, "to": decorator_name, "type": "decorator"})

        # ---------------------------------------
        # CALL EXPRESSIONS
        # ---------------------------------------
        if kind == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                func_name = get_text(code, func_node)
                edges.append({"from": file_id, "to": func_name, "type": "call"})

        if kind == "import_statement":
            src_node = node.child_by_field_name("source")
            if src_node:
                module_path = get_text(code, src_node).strip("\"'")
                edges.append({"from": file_id, "to": module_path, "type": "import"})

    return nodes, edges


def build_code_graph(root_dir):
    graph_nodes = {}
    graph_edges = []

    for root, _, files in os.walk(root_dir):
        print(root, files)
        for file in files:
            if not file.endswith((".ts", ".tsx")):
                continue

            path = os.path.join(root, file)
            nodes, edges = build_graph_for_file(path)

            graph_nodes.update(nodes)
            graph_edges.extend(edges)

    return {"nodes": graph_nodes, "edges": graph_edges}


if __name__ == "__main__":
    graph = build_code_graph("sample_data")
    with open("code_graph.json", "w") as f:
        json.dump(graph, f, indent=2)

    print("Graph saved → code_graph.json")
