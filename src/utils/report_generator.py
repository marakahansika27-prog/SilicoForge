import os
from typing import Dict, Any

class ReportGenerator:
    """Generates Markdown debug reports for each module."""
    def __init__(self, module_name: str, save_dir: str = "outputs/reports"):
        self.module_name = module_name
        self.save_dir = save_dir
        self.filepath = os.path.join(self.save_dir, f"{self.module_name}_REPORT.md")
        os.makedirs(self.save_dir, exist_ok=True)
        self.content = f"# {self.module_name} DEBUG REPORT\n\n"
        
    def add_section(self, title: str, content: str):
        self.content += f"## {title}\n{content}\n\n"

    def add_parameters(self, params: Dict[str, Any]):
        param_str = "\n".join([f"- **{k}**: {v}" for k, v in params.items()])
        self.add_section("Parameters Used", param_str)
        
    def add_execution_stats(self, time_ms: float, mem_kb: float, dims: str):
        stats = f"- **Runtime**: {time_ms:.2f} ms\n- **Memory Diff**: {mem_kb:.2f} KB\n- **Output Dimensions**: {dims}"
        self.add_section("Execution Statistics", stats)
        
    def add_images(self, image_paths: list[str]):
        img_str = "\n".join([f"![image]({path})" for path in image_paths])
        self.add_section("Generated Images", img_str)
        
    def add_status(self, passed: bool, observations: str):
        status = "PASS" if passed else "FAIL"
        content = f"**Status**: {status}\n\n**Observations**:\n{observations}"
        self.add_section("Conclusion", content)
        
    def save(self):
        with open(self.filepath, "w") as f:
            f.write(self.content)
