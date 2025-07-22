import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

class LoadModelConfig:
    def __init__(self, yaml_file: Union[str, Path] = None):
        """
        Inititalize the model config class

        Args:
            yaml_file: Path to the YAML configuration file
        """
        if yaml_file is None:
            yaml_file = Path(__file__).parent.parent / "config" / "model_config.yaml"

        self.yaml_file = Path(yaml_file)
        self.config_data = self.load_yaml()

    def load_yaml(self) -> Dict:
        """Load the YAML configuration file."""
        try:
            if not self.yaml_file.exists():
                raise FileNotFoundError(f"The file {self.yaml_file} does not exist.")
            
            with self.yaml_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("model_configs", {})
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise IOError(f"Error reading {self.yaml_file}: {e}")
        
    def list_all_models(self) -> list:
        """Return a list of all model keys."""
        return list(self.config_data.keys())
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract the model configuration for a given model name.

        Args:
            model_name: The name of the model to extract configuration for.

        Returns:
            Model Configuration dictionary or None if the model name is not found.
        """
        return self.config_data.get(model_name, None)
    
