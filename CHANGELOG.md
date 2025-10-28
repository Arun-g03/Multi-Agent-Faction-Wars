# Changelog

All notable changes to the Multi-Agent Faction Wars project will be documented in this file.

## [2025-10-27]

## Development in progress...

### Added
- **Peacekeeper Block Ability**
  - Added "block" action to peacekeeper actions list in `ROLE_ACTIONS_MAP`
  - Implemented defensive block behavior in `peacekeeper_behaviours.py`
  - Block function detects nearby threats (within 2 cells) and enters defensive stance
  - Added attack detection system: agents track `is_under_attack` when health decreases
  - Returns ONGOING when under attack or threats within 2 cells
  - Returns FAILURE when blocking unnecessarily (no active threat)
  - Attack state persists for 5 steps after taking damage
  - Added block-specific rewards: +0.15 bonus for peacekeepers using block action
  - Encourages defensive tactics only when actually threatened
  - Peacekeepers can now hold position and defend when attacked or in immediate danger
  - Added health maintenance bonus: peacekeepers with >70% health receive up to +0.09 bonus
  - Blocking helps maintain high health, leading to better long-term rewards and survival
  - Higher health = better performance = higher rewards over time

- **Gatherer Plant Tree Ability**
  - Added "plant_tree" action to gatherer actions list in `ROLE_ACTIONS_MAP`
  - Implemented tree planting behavior in `gatherer_behaviours.py`
  - Costs 3 food from faction to plant a new apple tree
  - Planted trees start as saplings (stage 0) and grow over 6 stages (5 minutes total)
  - Trees only produce apples when mature (stage 3+: produces 2-10 apples based on stage)
  - Natural/map-spawned trees start fully mature (stage 5)
  - Growth stages show visual progression using 6-frame sprite sheet
  - Sprite sheet is loaded once and cached for all trees (performance optimization)
  - Render method dynamically extracts correct growth stage sprite from cached sheet
  - Validates planting location (must be land, not water, not occupied)
  - Added plant-specific rewards: +0.3 bonus for successfully planting trees
  - Strategic timing: must wait for growth before harvesting
  - Saplings (stages 0-2) have 0 apples and cannot be foraged
  - Only trees at stage 3+ can be foraged for apples
  - Apple quantity scales with growth stage (stage 3=2, stage 4=4, stage 5=6-10 apples)
  - Fixed bug: planted saplings now correctly start with 0 apples instead of default spawn quantity
  - Strategic risk: planted trees can be foraged by any faction
  - Encourages gatherers to establish their own resource production
  - Resource expansion and territory control mechanic
  - Added "PLANT_TREES" as HQ strategy option
  - HQ can now select planting strategy to expand resource production
  - Rewards planting strategically: +1.2 for actively planting when room for expansion
  - Penalizes planting when no food or gatherers available: -0.5
  - Checks existing tree count to prevent excessive planting (max ~20 trees recommended)

- **Gatherer Plant Gold Vein Ability**
  - Added "plant_gold_vein" action to gatherer actions list
  - Implemented gold vein planting behavior in `gatherer_behaviours.py`
  - Costs 5 gold from faction to plant a new gold vein (expensive investment)
  - Gold vein yields random amount: 5-10 gold pieces with weighted distribution
  - 50% chance: 5-6 gold (break even or small profit)
  - 30% chance: 7-8 gold (moderate profit)
  - 20% chance: 9-10 gold (rare big win!)
  - Profitable economy: spend 5 to get 5-10 back (guaranteed at least break even)
  - Validates planting location (must be land, not water, not occupied)
  - Added "PLANT_GOLD_VEINS" as HQ strategy option
  - HQ can now select gold planting strategy for economic expansion
  - Strategic requirement: Must have at least 20 gold reserve to plant veins (conserves precious resources)
  - Rewards planting strategically: +1.5 for actively planting with <10 veins and spare gold
  - Penalizes wasting gold: -0.3 when planting without sufficient reserves
  - Gold economy now makes sense: expensive to produce but valuable to have

### Fixed
- **Profiling Error Handling**
  - Fixed `UnboundLocalError` in `profile_function()` when SystemExit occurred
  - Initialize `result = None` before try block to prevent unbound variable error
  - Allow SystemExit and KeyboardInterrupt to propagate normally (don't swallow them)
  - Improved exception handling to properly save profiling results even on exit

### Changed
- **Resource Classes Refactoring**
  - Created new `ENVIRONMENT/Resources/` directory to organize resource classes
  - Moved `AppleTree` class to `ENVIRONMENT/Resources/apple_tree.py`
  - Moved `GoldLump` class to `ENVIRONMENT/Resources/gold_lump.py`
  - Updated all imports across codebase to use `from ENVIRONMENT.Resources import AppleTree, GoldLump`
  - Created `ENVIRONMENT/Resources/__init__.py` for clean module imports
  - Improved code organization and separation of concerns
  - Makes it easier to add new resource types in the future