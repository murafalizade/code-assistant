from tree_sitter import Language, Parser
import tree_sitter_typescript as tstypescript

class CodeChunkExtractor:
    def __init__(self, code):
        self.chunks = []
        self.code_lines = code.split("\n")
        self.code_bytes = code.encode("utf8")
        lang_grammar = tstypescript.language_typescript()
        self.parser = Parser(Language(lang_grammar))
        
        self.tree = self.parser.parse(self.code_bytes)

    def get_text(self, node):
        start = node.start_point[0]
        end = node.end_point[0]
        return "\n".join(self.code_lines[start : end + 1])

    def get_name(self, node):
        for child in node.children:
            if child.type in ("identifier", "property_identifier", "type_identifier"):
                return self.code_bytes[child.start_byte:child.end_byte].decode('utf8')
        return None

    def get_chunk(self, node):
        return {
            "text": self.get_text(node),
            "node_type": node.type,
            "name": self.get_name(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }
    
    def walk(self, node):
            relevant_types = [
                "class_declaration",
                "function_declaration",
                "method_definition",
                "arrow_function",
                "function_signature",
            ]

            if node.type in relevant_types:
                self.chunks.append(self.get_chunk(node))

            for c in node.children:
                self.walk(c)
    
    def get_chunks(self):
        self.walk(self.tree.root_node)
        return self.chunks
