# Drift-Sense V2 Standard Module Interface

All engines implement the following standard API:

```python
class BaseEngine:
    def __init__(self, **kwargs):
        # Initialize engine parameters
        self.stats = {}

    def run(self, inputs: dict) -> dict:
        """
        Executes the engine logic.
        
        Args:
            inputs (dict): A dictionary containing required inputs.
            
        Returns:
            dict: A dictionary containing outputs and a 'stats' dict with profiling info.
        """
        pass
```
