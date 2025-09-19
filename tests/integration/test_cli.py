"""
Basic integration test for the CLI functionality
"""

import subprocess
import tempfile
import os
from pathlib import Path

def test_cli_help():
    """Test that the main CLI shows help."""
    result = subprocess.run(["llmctl", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Distributed LLM Training and Inference System" in result.stdout

def test_hw_probe():
    """Test hardware probing functionality."""
    result = subprocess.run(["llmctl", "hw", "probe"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hardware Profile" in result.stdout

def test_init_scaffold():
    """Test project scaffolding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run([
            "llmctl", "init", "scaffold", 
            "--template", "gpt", 
            "--size", "7b",
            "--name", "test-proj",
            "--output-dir", tmpdir
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Check that files were created
        project_dir = Path(tmpdir) / "test-proj"
        assert project_dir.exists()
        assert (project_dir / "configs" / "models" / "gpt-7b.json").exists()
        assert (project_dir / "configs" / "default.toml").exists()
        assert (project_dir / "README.md").exists()

def test_plan_workflow():
    """Test the complete plan workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First scaffold a project
        subprocess.run([
            "llmctl", "init", "scaffold",
            "--template", "gpt",
            "--size", "7b", 
            "--name", "test-proj",
            "--output-dir", tmpdir
        ], capture_output=True, text=True)
        
        project_dir = Path(tmpdir) / "test-proj"
        os.chdir(project_dir)
        
        # Generate hardware profile
        result = subprocess.run([
            "llmctl", "hw", "probe", 
            "--emit", "configs/hw/local.toml"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert (project_dir / "configs" / "hw" / "local.toml").exists()
        
        # Compute plan
        result = subprocess.run([
            "llmctl", "plan", "compute",
            "--model", "configs/models/gpt-7b.json",
            "--hardware", "configs/hw/local.toml",
            "--out", "plans/local.toml"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert (project_dir / "plans" / "local.toml").exists()

if __name__ == "__main__":
    test_cli_help()
    test_hw_probe()
    test_init_scaffold()
    test_plan_workflow()
    print("All tests passed!")