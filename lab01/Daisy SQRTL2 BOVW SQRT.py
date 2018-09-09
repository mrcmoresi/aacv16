Daisy SQRTL2 BOVW SQRT
params_C = [
0.001: [0.6766666666666666, 0.6666666666666666, 0.72, 0.6433333333333333, 0.6966666666666667]
0.01: [0.6233333333333333, 0.6333333333333333, 0.6466666666666666, 0.6133333333333333, 0.6366666666666667],
0.1: [0.5566666666666666, 0.56, 0.6266666666666667, 0.5333333333333333, 0.56],
1.0: [0.53, 0.5466666666666666, 0.58, 0.5066666666666667, 0.5233333333333333],
10.0: [0.53, 0.5466666666666666, 0.5666666666666667, 0.5, 0.5],
100.0: [0.55, 0.5466666666666666, 0.5533333333333333, 0.5066666666666667, 0.5],
1000.0: [0.5566666666666666, 0.53, 0.5733333333333334, 0.5066666666666667, 0.5066666666666667]
]

list_median = {} [0.67666666666666664, 0.6333333333333333, 0.56000000000000005, 0.53000000000000003, 0.53000000000000003, 0.54666666666666663, 0.53000000000000003]

C = 0.001


##############################################################

accuracy_list = [0.6787269681742043, 0.685427135678392, 0.6907872696817421, 0.6911222780569515, 0.6857621440536014, 0.6850921273031826, 0.6901172529313233, 0.6948073701842546, 0.683752093802345, 0.685427135678392, 0.6800670016750419, 0.697822445561139, 0.6800670016750419, 0.6850921273031826, 0.6824120603015076, 0.6931323283082077, 0.6871021775544388, 0.6934673366834171, 0.685427135678392, 0.68107202680067, 0.6814070351758794, 0.6917922948073701, 0.6814070351758794, 0.6924623115577889, 0.6894472361809045, 0.6938023450586265, 0.6793969849246231, 0.6931323283082077, 0.6787269681742043, 0.6747068676716917, 0.6964824120603015, 0.6938023450586265, 0.6927973199329983, 0.6891122278056951, 0.6804020100502512, 0.6887772194304858, 0.6887772194304858, 0.6901172529313233, 0.6864321608040201, 0.6907872696817421, 0.6934673366834171, 0.6988274706867672, 0.688107202680067, 0.6924623115577889, 0.6850921273031826, 0.7001675041876047, 0.6864321608040201, 0.6790619765494137, 0.6894472361809045, 0.6938023450586265, 0.685427135678392, 0.6921273031825795, 0.6760469011725293, 0.6907872696817421, 0.6780569514237856, 0.6917922948073701, 0.6850921273031826, 0.6924623115577889, 0.6860971524288108, 0.6864321608040201, 0.6917922948073701, 0.6921273031825795, 0.6753768844221105, 0.6820770519262982, 0.685427135678392, 0.678391959798995, 0.6877721943048576, 0.6864321608040201, 0.6914572864321608, 0.6767169179229481, 0.6971524288107203, 0.6871021775544388, 0.6763819095477387, 0.6871021775544388, 0.6938023450586265, 0.6988274706867672, 0.683752093802345, 0.7025125628140704, 0.6857621440536014, 0.6844221105527638, 0.6874371859296482, 0.6733668341708543, 0.6907872696817421, 0.6847571189279732, 0.6927973199329983, 0.6884422110552764, 0.7011725293132328, 0.6847571189279732, 0.6834170854271356, 0.6901172529313233, 0.6961474036850921, 0.6884422110552764, 0.6944723618090453, 0.6884422110552764, 0.6857621440536014, 0.6807370184254606, 0.682747068676717, 0.6874371859296482, 0.6894472361809045, 0.6961474036850921, 0.6844221105527638, 0.685427135678392, 0.6921273031825795, 0.68107202680067, 0.6887772194304858, 0.6904522613065327, 0.6864321608040201, 0.6941373534338359, 0.6968174204355109, 0.6877721943048576, 0.6696817420435511, 0.68107202680067, 0.6934673366834171, 0.6874371859296482, 0.6884422110552764, 0.68107202680067, 0.699497487437186, 0.6877721943048576, 0.6921273031825795, 0.6757118927973199, 0.6824120603015076, 0.6877721943048576, 0.6931323283082077, 0.6998324958123953, 0.6904522613065327, 0.6971524288107203, 0.6830820770519263, 0.6927973199329983, 0.6817420435510888, 0.6737018425460637, 0.685427135678392, 0.6817420435510888, 0.6860971524288108, 0.6974874371859296, 0.6867671691792295, 0.6907872696817421, 0.688107202680067, 0.6817420435510888, 0.6981574539363484, 0.6807370184254606, 0.6860971524288108, 0.6938023450586265, 0.6948073701842546, 0.6887772194304858, 0.6773869346733669, 0.6793969849246231, 0.6954773869346733, 0.6804020100502512, 0.6938023450586265, 0.7011725293132328, 0.6871021775544388, 0.6847571189279732, 0.6857621440536014, 0.6968174204355109, 0.6844221105527638, 0.6924623115577889, 0.6954773869346733, 0.6904522613065327, 0.678391959798995, 0.6958123953098827, 0.697822445561139, 0.6847571189279732, 0.6901172529313233, 0.6941373534338359, 0.6971524288107203, 0.6807370184254606, 0.6911222780569515, 0.6917922948073701, 0.685427135678392, 0.6740368509212731, 0.695142378559464, 0.6884422110552764, 0.6860971524288108, 0.6887772194304858, 0.6901172529313233, 0.6948073701842546, 0.6927973199329983, 0.6931323283082077, 0.6897822445561139, 0.6938023450586265, 0.685427135678392, 0.6824120603015076, 0.6844221105527638, 0.6934673366834171, 0.6974874371859296, 0.6807370184254606, 0.6743718592964824, 0.6847571189279732, 0.6921273031825795, 0.6847571189279732, 0.6847571189279732, 0.6917922948073701, 0.6860971524288108, 0.6857621440536014, 0.685427135678392, 0.6941373534338359, 0.6800670016750419, 0.6971524288107203, 0.6938023450586265, 0.6850921273031826]
media accuracy = 0.6879
Desviacion standard = 0.0064 accuracy = 0.6879