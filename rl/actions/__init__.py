from rl.actions.add_direction_of_change import AddDirectionOfChange
from rl.actions.add_missingness_indicator import AddMissingnessIndicator
from rl.actions.refine_granularity import RefineGranularity
from rl.actions.remove_lowest_variance_field import RemoveLowestVarianceField
from rl.actions.remove_redundant_pair import RemoveRedundantPair
from rl.actions.split_highest_entropy_field import SplitHighestEntropyField

ALL_ACTIONS = [
    RemoveLowestVarianceField(),
    AddDirectionOfChange(),
    AddMissingnessIndicator(),
    SplitHighestEntropyField(),
    RemoveRedundantPair(),
    RefineGranularity(),
]
