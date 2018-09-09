Daisy L2 Bovw SQRTL2
params_C = [
0.001: [0.52, 0.5866666666666667, 0.5733333333333334, 0.5166666666666667, 0.5833333333333334],
0.01: [0.6233333333333333, 0.6533333333333333, 0.68, 0.58, 0.66],
0.1: [0.6566666666666666, 0.7166666666666667, 0.7233333333333334, 0.61, 0.7233333333333334],
1.0: [0.6166666666666667, 0.7, 0.7166666666666667, 0.6033333333333334, 0.7566666666666667],
10.0: [0.58, 0.63, 0.6466666666666666, 0.55, 0.69],
100.0: [0.5433333333333333, 0.5966666666666667, 0.5833333333333334, 0.5133333333333333, 0.5933333333333334],
1000.0: [0.5266666666666666, 0.5566666666666666, 0.57, 0.52, 0.63]
]
list_median = [0.57333333333333336, 0.65333333333333332, 0.71666666666666667, 0.69999999999999996, 0.63, 0.58333333333333337, 0.55666666666666664]


C = 0.1

#############################################################
accuracy_list = [0.6850921273031826, 0.6927973199329983, 0.6887772194304858, 0.7018425460636516, 0.6984924623115578, 0.6907872696817421, 0.6941373534338359, 0.7041876046901172, 0.6944723618090453, 0.700502512562814, 0.6958123953098827, 0.6988274706867672, 0.7055276381909548, 0.6998324958123953, 0.6924623115577889, 0.6887772194304858, 0.6934673366834171, 0.7035175879396985, 0.6891122278056951, 0.6867671691792295, 0.6877721943048576, 0.6917922948073701, 0.685427135678392, 0.6944723618090453, 0.6984924623115578, 0.6904522613065327, 0.7035175879396985, 0.6904522613065327, 0.6917922948073701, 0.6981574539363484, 0.6988274706867672, 0.7075376884422111, 0.700502512562814, 0.6927973199329983, 0.6901172529313233, 0.6894472361809045, 0.697822445561139, 0.7008375209380234, 0.6998324958123953, 0.6971524288107203, 0.6901172529313233, 0.6961474036850921, 0.7028475711892798, 0.6971524288107203, 0.7031825795644892, 0.700502512562814, 0.6948073701842546, 0.7011725293132328, 0.6897822445561139, 0.7001675041876047, 0.7008375209380234, 0.7028475711892798, 0.6927973199329983, 0.6887772194304858, 0.6988274706867672, 0.7001675041876047, 0.7015075376884422, 0.6901172529313233, 0.6964824120603015, 0.6961474036850921, 0.6968174204355109, 0.6968174204355109, 0.6988274706867672, 0.6871021775544388, 0.6924623115577889, 0.6988274706867672, 0.6954773869346733, 0.6877721943048576, 0.700502512562814, 0.7018425460636516, 0.6877721943048576, 0.6934673366834171, 0.6964824120603015, 0.6917922948073701, 0.7072026800670017, 0.6864321608040201, 0.7068676716917923, 0.697822445561139, 0.6998324958123953, 0.6901172529313233, 0.7038525963149078, 0.6941373534338359, 0.6988274706867672, 0.7001675041876047, 0.6948073701842546, 0.6964824120603015, 0.6964824120603015, 0.6984924623115578, 0.7028475711892798, 0.7011725293132328, 0.6938023450586265, 0.6907872696817421, 0.6884422110552764, 0.6911222780569515, 0.700502512562814, 0.7008375209380234, 0.7061976549413735, 0.6968174204355109, 0.6971524288107203, 0.7001675041876047, 0.7011725293132328, 0.6954773869346733, 0.697822445561139, 0.6934673366834171, 0.697822445561139, 0.6961474036850921, 0.6907872696817421, 0.6958123953098827, 0.6938023450586265, 0.7045226130653266, 0.6981574539363484, 0.7011725293132328, 0.6954773869346733, 0.6961474036850921, 0.6894472361809045, 0.6954773869346733, 0.7061976549413735, 0.6971524288107203, 0.7011725293132328, 0.6998324958123953, 0.6958123953098827, 0.6961474036850921, 0.6887772194304858, 0.6927973199329983, 0.702177554438861, 0.7015075376884422, 0.6911222780569515, 0.6998324958123953, 0.7061976549413735, 0.6981574539363484, 0.697822445561139, 0.7008375209380234, 0.6991624790619766, 0.6958123953098827, 0.6971524288107203, 0.6944723618090453, 0.6941373534338359, 0.7008375209380234, 0.6907872696817421, 0.6934673366834171, 0.6931323283082077, 0.700502512562814, 0.6964824120603015, 0.6891122278056951, 0.6911222780569515, 0.697822445561139, 0.6917922948073701, 0.6974874371859296, 0.6954773869346733, 0.6958123953098827, 0.6948073701842546, 0.6948073701842546, 0.702177554438861, 0.6961474036850921, 0.7018425460636516, 0.697822445561139, 0.6984924623115578, 0.695142378559464, 0.7055276381909548, 0.704857621440536, 0.6954773869346733, 0.6961474036850921, 0.688107202680067, 0.6874371859296482, 0.6938023450586265, 0.6948073701842546, 0.6971524288107203, 0.6894472361809045, 0.6988274706867672, 0.6944723618090453, 0.6927973199329983, 0.7008375209380234, 0.6897822445561139, 0.6998324958123953, 0.6954773869346733, 0.6894472361809045, 0.6941373534338359, 0.6894472361809045, 0.6998324958123953, 0.6884422110552764, 0.6914572864321608, 0.6941373534338359, 0.6877721943048576, 0.700502512562814, 0.7018425460636516, 0.6968174204355109, 0.699497487437186, 0.6961474036850921, 0.68107202680067, 0.6871021775544388, 0.6971524288107203, 0.6901172529313233, 0.6934673366834171, 0.6847571189279732, 0.6938023450586265, 0.6958123953098827, 0.6904522613065327, 0.6941373534338359, 0.688107202680067, 0.6941373534338359]
media accuracy = 0.6961
Desviacion standard = 0.0051 accuracy = 0.6961


