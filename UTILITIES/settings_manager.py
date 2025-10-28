"""
Settings Manager - Handles persistent settings storage in JSON format
"""

import json
import os
from pathlib import Path


class SettingsManager:
    """Manages persistent application settings stored in JSON format"""

    def __init__(self, settings_file="settings.json"):
        """
        Initialize the settings manager.

        Args:
            settings_file: Name of the settings file (stored in project root)
        """
        self.project_root = Path(__file__).parent.parent
        self.settings_file = self.project_root / settings_file
        self.settings = self._load_settings()

    def _get_default_settings(self):
        """Return default settings"""
        return {
            "setup": {"has_run_installer": False, "first_run": True},
            "runtime": {
                "headless_mode": False,
                "last_episodes": 30,
                "last_steps_per_episode": 5000,
            },
            "display": {"last_screen_width": 1280, "last_screen_height": 720},
            "training": {"enable_tensorboard": False, "enable_plots": False},
        }

    def _load_settings(self):
        """Load settings from JSON file, create with defaults if not exists"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                default_settings = self._get_default_settings()
                settings = self._merge_settings(default_settings, settings)
                return settings
            except Exception as e:
                print(f"[WARNING] Failed to load settings: {e}")
                return self._get_default_settings()
        else:
            # First run - create default settings file
            default_settings = self._get_default_settings()
            self._save_settings(default_settings)
            return default_settings

    def _merge_settings(self, defaults, loaded):
        """Recursively merge loaded settings with defaults"""
        merged = defaults.copy()
        for key, value in loaded.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_settings(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _save_settings(self, settings=None):
        """Save settings to JSON file"""
        settings_to_save = settings if settings is not None else self.settings
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings_to_save, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Failed to save settings: {e}")

    def get(self, category, key, default=None):
        """Get a setting value by category and key"""
        return self.settings.get(category, {}).get(key, default)

    def set(self, category, key, value):
        """Set a setting value by category and key"""
        if category not in self.settings:
            self.settings[category] = {}
        self.settings[category][key] = value
        self._save_settings()

    def get_all(self):
        """Get all settings"""
        return self.settings

    def update(self, updates):
        """Update multiple settings at once"""
        for category, values in updates.items():
            if category not in self.settings:
                self.settings[category] = {}
            self.settings[category].update(values)
        self._save_settings()

    # Convenience methods for common settings
    def has_run_installer(self):
        """Check if the installer has been run"""
        return self.get("setup", "has_run_installer", False)

    def set_installer_run(self, value=True):
        """Mark the installer as run"""
        self.set("setup", "has_run_installer", value)

    def is_first_run(self):
        """Check if this is the first run"""
        return self.get("setup", "first_run", True)

    def set_first_run(self, value=False):
        """Mark first run as complete"""
        self.set("setup", "first_run", value)

    def is_headless_mode(self):
        """Get headless mode setting"""
        return self.get("runtime", "headless_mode", False)

    def set_headless_mode(self, value):
        """Set headless mode"""
        self.set("runtime", "headless_mode", value)

    def force_reinstall(self):
        """Mark installer as needing to re-run (for dependency error recovery)"""
        self.set("setup", "has_run_installer", False)
        self.set("setup", "force_reinstall", True)

    def should_force_reinstall(self):
        """Check if we should force reinstall"""
        return self.get("setup", "force_reinstall", False)

    def clear_force_reinstall(self):
        """Clear the force reinstall flag"""
        self.set("setup", "force_reinstall", False)

    def get_last_episodes(self):
        """Get last used episode count"""
        return self.get("runtime", "last_episodes", 30)

    def set_last_episodes(self, value):
        """Set last used episode count"""
        self.set("runtime", "last_episodes", value)

    def get_last_steps(self):
        """Get last used steps per episode"""
        return self.get("runtime", "last_steps_per_episode", 5000)

    def set_last_steps(self, value):
        """Set last used steps per episode"""
        self.set("runtime", "last_steps_per_episode", value)

    def update_config_settings(self, settings_dict):
        """Update multiple config settings at once"""
        if "config" not in self.settings:
            self.settings["config"] = {}
        self.settings["config"].update(settings_dict)
        self._save_settings()

    def get_config_settings(self):
        """Get all config settings"""
        return self.settings.get("config", {})

    def get_config_setting(self, key, default=None):
        """Get a specific config setting"""
        return self.settings.get("config", {}).get(key, default)


# Global settings manager instance
settings_manager = SettingsManager()
