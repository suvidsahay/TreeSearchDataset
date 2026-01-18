import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import networkx as nx
from pyvis.network import Network
import re

# --- FEATURE: PrettyPrinter Class for clean logging ---
class PrettyPrinter:
    PURPLE = '\033[95m'; CYAN = '\033[96m'; BLUE = '\033[94m'; GREEN = '\033[92m'
    YELLOW = '\033[93m'; RED = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'
    
    def header(self, text): print(f"\n{self.PURPLE}{self.BOLD}{'=' * 25} {text} {'=' * 25}{self.ENDC}")
    def info(self, text, indent=0): print(f"{' ' * indent}{self.BLUE}INFO: {text}{self.ENDC}")
    def success(self, text, indent=0): print(f"{' ' * indent}{self.GREEN}✅ {text}{self.ENDC}")
    def warning(self, text, indent=0): print(f"{' ' * indent}{self.YELLOW}⚠️ {text}{self.ENDC}")
    def step(self, text, indent=0): print(f"{' ' * indent}{self.CYAN}➡️  {text}{self.ENDC}")
    
    def print_question_state(self, state, indent=0):
        passage_titles = [f"'{title}'" for title, _ in state.passages_used]
        print(f"{' ' * indent}❓ {self.BOLD}Q:{self.ENDC} \"{state.question}\"")
        print(f"{' ' * (indent+2)}{self.YELLOW}Hops:{self.ENDC} {len(state.passages_used)} | {self.YELLOW}Passages:{self.ENDC} {', '.join(passage_titles)}")
    
    def print_pqs_debug(self, pqs: List[List[tuple]]):
        """Visually prints the state of all priority queues for debugging."""
        print("\n" + "=" * 25 + " DEBUG: PRIORITY QUEUE STATE " + "=" * 25)
        for i, pq in enumerate(pqs):
            print(f"\n--- PQ {i} (Contains {i + 1}-hop questions)")
            if not pq:
                print("[EMPTY]")
                continue
            sorted_pq = sorted(pq, key=lambda x: x[0], reverse=False)
            for neg_score, _, state, passage_to_add in sorted_pq:
                score = -neg_score
                question_preview = state.question
                print(f"  - Candidate Score: {score:.4f}")
                print(f"    - Current Question: \"{question_preview}\"")
                print(f"    - Passages Used: {[p[0] for p in state.passages_used]}")
                print(f"    - -> Next Passage to Add: '{passage_to_add[0]}'")
        print("=" * 75 + "\n")


pp = PrettyPrinter()


# --- Feature: Tree Visualization Structures ---

@dataclass
class HistoryNode:
    """Represents a generated question node in the expansion tree."""
    id: str  # Unique ID for the question (e.g., Q1, Q2, Q3...)
    parent_ids: List[str]  # List of IDs of nodes this question was built from
    question_text: str
    answer: str
    explanation: str
    passages_used: List[Tuple[str, str]]
    is_seed: bool
    verification_status: str = "unknown"  # 'passed', 'failed', 'intermediate'

    @property
    def hop_count(self):
        return len(self.passages_used)


# Define a global counter for unique question IDs
QUESTION_ID_COUNTER = itertools.count(1)


def render_quest_tree_dot(claim: str, history: List[HistoryNode], output_filename: str):
    """
    Renders the question expansion history as a Graphviz (DOT) file.
    Color codes nodes based on verification status.
    """
    node_definitions = set()
    edge_definitions = []

    # Map to ensure Passage nodes are only defined once
    passage_nodes = {}

    # Function to sanitize text for DOT/HTML labels
    def sanitize(text):
        if text is None: return "N/A"
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '')

    # ID Sanitizer
    def sanitize_id(title, length=20):
        title = title.replace(' ', '_').replace('-', '_').replace('.', '_')
        title = re.sub(r'[^a-zA-Z0-9_]', '', title)
        return title[:length]

    # Text Wrapper
    def wrap_text(text, limit=100):
        if text is None: return "N/A"
        words = text.split()
        if not words: return ""
        wrapped_lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > limit:
                wrapped_lines.append(current_line.strip())
                current_line = word + " "
            else:
                current_line += word + " "
        wrapped_lines.append(current_line.strip())
        return "<BR/>".join(wrapped_lines)

    # 1. Define nodes and edges
    for node in history:
        # ------------------------------------
        # A. Define Question Node (Q-node)
        # ------------------------------------

        # Determine Color based on Status
        if node.verification_status == "passed":
            fillcolor = "lightgreen" # Success
            border_color = "darkgreen"
        elif node.verification_status == "failed":
            fillcolor = "#FFCDD2" # Light Red
            border_color = "red"
        else:
            fillcolor = "lightblue" # Intermediate / Unknown
            border_color = "blue"

        q_label_html = f'''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" COLOR="{border_color}">
          <TR><TD COLSPAN="2" BGCOLOR="{fillcolor}"><B>Q{node.id} ({node.hop_count} Hops) [{node.verification_status.upper()}]</B></TD></TR>
          <TR><TD ALIGN="LEFT"><B>Question:</B></TD><TD ALIGN="LEFT">{wrap_text(sanitize(node.question_text))}</TD></TR>
          <TR><TD ALIGN="LEFT"><B>Answer:</B></TD><TD ALIGN="LEFT">{wrap_text(sanitize(node.answer))}</TD></TR>
          <TR><TD ALIGN="LEFT"><B>Explanation:</B></TD><TD ALIGN="LEFT">{wrap_text(sanitize(node.explanation))}</TD></TR>
        </TABLE>>'''

        q_node_id = f"Q_{node.id}"
        
        node_definitions.add(
            f'{q_node_id} [label={q_label_html}, shape=box, style="filled, rounded", fillcolor="{fillcolor}", color="{border_color}"];')

        # ------------------------------------
        # B. Define Passage Nodes (P-nodes) and Edges
        # ------------------------------------

        if node.is_seed:
            passage_title, passage_text = node.passages_used[0]
            p_text_wrapped = wrap_text(sanitize(passage_text))

            p_label_html = f'''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
              <TR><TD COLSPAN="1" BGCOLOR="bisque"><B>PASSAGE SOURCE:</B> {sanitize(passage_title)}</TD></TR>
              <TR><TD ALIGN="LEFT">{p_text_wrapped}</TD></TR>
            </TABLE>>'''

            p_id = f"P_{sanitize_id(passage_title)}"

            if p_id not in passage_nodes:
                node_definitions.add(f'{p_id} [label={p_label_html}, shape=box, style=filled, fillcolor="bisque"];')
                passage_nodes[p_id] = passage_title

            # Edge: Passage -> Question
            edge_definitions.append(f'{p_id} -> {q_node_id} [label="seed question"];')

        else:
            # 1. Edge from Parent Question
            if node.parent_ids:
                parent_q_node_id = f"Q_{node.parent_ids[0]}"
                edge_definitions.append(f'{parent_q_node_id} -> {q_node_id} [label="expands on Q{node.parent_ids[0]}"];')

            # 2. Edge from New Passage
            passage_title, passage_text = node.passages_used[-1]
            p_text_wrapped = wrap_text(sanitize(passage_text))

            p_label_html = f'''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
              <TR><TD COLSPAN="1" BGCOLOR="bisque"><B>PASSAGE SOURCE:</B> {sanitize(passage_title)}</TD></TR>
              <TR><TD ALIGN="LEFT">{p_text_wrapped}</TD></TR>
            </TABLE>>'''

            sanitized_title = sanitize_id(passage_title)
            new_passage_id = f"P_{sanitized_title}_H{node.hop_count}"

            if new_passage_id not in passage_nodes:
                node_definitions.add(
                    f'{new_passage_id} [label={p_label_html}, shape=box, style=filled, fillcolor="bisque"];')
                passage_nodes[new_passage_id] = passage_title

            edge_definitions.append(f'{new_passage_id} -> {q_node_id} [label="new fact source", style=dashed];')

    # 2. Compile DOT output
    dot_output = [
        f'digraph QuestTree_{claim.replace(" ", "_")[:20]} {{',
        '  rankdir=LR;',
        '  node [fontsize=10];',
        '  overlap=false;',
        '  splines=true;'
    ]
    dot_output.extend(sorted(list(node_definitions)))
    dot_output.extend(edge_definitions)
    dot_output.append('}')

    # 3. Write to file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dot_output))

    pp.success(f"Tree structure saved in DOT format: {output_filename}", indent=2)


def render_quest_tree_html(claim: str, history: List[HistoryNode], output_filename="quest_tree_interactive.html"):
    """
    Renders the question expansion history as an interactive HTML file using Pyvis.
    Color codes nodes based on verification status.
    """

    G = nx.DiGraph()
    WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"

    # 1. Initialize Pyvis Network
    net = Network(height="800px", width="100%", notebook=False, directed=True,
                  heading=f"Interactive Expansion Tree for Claim: {claim}")

    # 2. Define Helper Functions
    def get_node_details(node: HistoryNode, type: str):
        detail_html = (
            f"<b>{type.upper()} NODE</b><hr>"
            f"<b>Status:</b> {node.verification_status.upper()}<br>"
            f"<b>Hops:</b> {node.hop_count}<br>"
            f"<b>Question:</b> {node.question_text}<br>"
            f"<b>Answer:</b> {node.answer}<br>"
            f"<b>Explanation:</b> {node.explanation}<br><br>"
            f"<b>Passages Used:</b> {', '.join([t for t, p in node.passages_used])}"
        )
        
        status_icon = "✅" if node.verification_status == "passed" else "❌" if node.verification_status == "failed" else "🔄"
        visible_label = f"{status_icon} Q{node.id}: {node.question_text[:30]}..."
        return visible_label, detail_html

    passage_nodes = {}

    for node in history:
        # A. Add Question Node (Q-node)
        q_label, q_title = get_node_details(node, "Question")
        q_node_id = f"Q_{node.id}"
        
        # Color Logic
        if node.verification_status == "passed":
            color = "#4CAF50"  # Green
        elif node.verification_status == "failed":
            color = "#F44336"  # Red
        else:
            color = "#2196F3"  # Blue

        G.add_node(q_node_id,
                   title=q_title,
                   label=q_label,
                   color=color,
                   shape='box')

        # B. Define Dependencies
        if node.is_seed:
            passage_title, passage_text = node.passages_used[0]
            p_node_id = f"P_{passage_title.replace(' ', '_')[:20]}"
            wiki_link = WIKI_BASE_URL + passage_title.replace(' ', '_')

            if p_node_id not in passage_nodes:
                passage_nodes[p_node_id] = passage_title
                p_title_html = (
                    f"<b>PASSAGE SOURCE:</b> <a href='{wiki_link}' target='_blank'>{passage_title}</a><hr>"
                    f"<b>Full Text:</b> {passage_text[:300]}..."
                )
                G.add_node(p_node_id,
                           title=p_title_html,
                           label=f"Passage: {passage_title[:20]}...",
                           color='bisque',
                           shape='ellipse',
                           href=wiki_link,
                           physics=False)

            G.add_edge(p_node_id, q_node_id, label="seed question")

        else:
            if node.parent_ids:
                parent_q_node_id = f"Q_{node.parent_ids[0]}"
                G.add_edge(parent_q_node_id, q_node_id, label="expands context")

            new_passage_title, new_passage_text = node.passages_used[-1]
            new_passage_id = f"P_{new_passage_title.replace(' ', '_')[:20]}_H{node.hop_count}"
            wiki_link = WIKI_BASE_URL + new_passage_title.replace(' ', '_')

            if new_passage_id not in passage_nodes:
                passage_nodes[new_passage_id] = new_passage_title
                p_title_html = (
                    f"<b>PASSAGE SOURCE:</b> <a href='{wiki_link}' target='_blank'>{new_passage_title}</a><hr>"
                    f"<b>Full Text:</b> {new_passage_text[:300]}..."
                )
                G.add_node(new_passage_id,
                           title=p_title_html,
                           label=f"Passage: {new_passage_title[:20]}...",
                           color='bisque',
                           shape='ellipse',
                           href=wiki_link,
                           physics=False)

            G.add_edge(new_passage_id, q_node_id, label="new fact source", dashes=True)

    # 4. Save to HTML
    net.from_nx(G)
    net.set_options("""
        {
          "interaction": {
            "hover": true,
            "tooltipDelay": 200
          },
          "htmlTitles": true,
          "configure": {
            "enabled": true,
            "filter": ["physics"]
          }
        }
    """)

    net.save_graph(output_filename)
    print(f"Interactive HTML graph saved to: {output_filename}")


def generate_output(claim: str, expansion_history: List[HistoryNode]):
    safe_claim = claim.replace(' ', '_').replace('/', '_')[:20]
    html_output_filename = f"quest_tree_{safe_claim}.html"
    dot_output_filename = f"quest_tree_{safe_claim}.dot"
    
    render_quest_tree_html(claim, expansion_history, html_output_filename)
    render_quest_tree_dot(claim, expansion_history, dot_output_filename)