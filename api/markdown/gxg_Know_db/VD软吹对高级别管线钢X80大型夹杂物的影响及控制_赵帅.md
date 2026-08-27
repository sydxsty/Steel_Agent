# VD 软吹对高级别管线钢X80 大型夹杂物的影响及控制

赵 帅1，2， 赵定国1， 李继新2， 钱云强2， 王书桓1（1. 华北理工大学冶金与能源学院， 河北 唐山 063009；2. 首钢京唐钢铁联合有限责任公司钢轧作业部， 河北 唐山 063210）

摘　要： 在高级别管线钢X80 冶炼过程中，大型夹杂物的控制是确保其性能稳定与工程适用性的关键因素之一。通过在LF（ladle furnace）-VD（vacuum degasser）（钢包炉-真空脱气炉）精炼流程中密集取样和系统分析，发现管线钢X80 的大型夹杂物集中出现在VD 钙处理后，主要类型为C A（12CaO·7Al O ）和CA（CaO·3Al O ），在软吹阶段未有效去除，大型夹杂物数量反而呈升高趋势，因此判断当前VD 软吹工艺控制不合理，导致夹杂物因搅拌动力不足而聚集长大形成大型夹杂物。结合钢包设备参数和工艺控制标准，对管线钢VD 软吹阶段进行了三维多物理场数值模拟，从流场分布、夹杂物迁移路径与钢液搅拌强度等方面进行全面分析，指出原软吹工艺在当前冶金反应条件下无法形成有效的夹杂物迁移通道；结合模拟结果，提出了新的VD 软吹流量控制方案，软吹流量由50 L/min增加至60 L/min，在确保钢液温度均匀性的同时，显著增强了钢液内部对流强度和夹杂物上浮效率。经工业试验验证，实施优化后的软吹工艺后，软吹结束钢液中大型夹杂物数量密度由0. 074 个/mm2降低至0. 020 个/mm2，板坯中大型夹杂物数量密度由0. 071 个/mm2降低至0. 040 个/mm2，显著提高了板坯的钢液质量和探伤合格率。该研究成果为高级别管线钢X80 在LF-VD 精炼流程中实现夹杂物精细控制提供了有效技术路径，也为后续其他高端钢种的洁净冶炼工艺优化提供了借鉴。

关键词： 管线钢； 高级别； 大型夹杂物； LF-VD 精炼工艺； 软吹流量； 数值模拟； 工业试验； 探伤文献标志码： A 文章编号：0449-749X（2025）10-0082-10

# Influence and control of VD soft blowing on large inclusions in high-grade pipeline steel X80

ZHAO Shuai1，2， ZHAO Dingguo1， LI Jixin2， QIAN Yunqiang2， WANG Shuhuan1（1. College of Metallurgy and Energy， North China University of Science and Technology，Tangshan 063009， Hebei， China； 2. Steel Rolling Operations Department， Shougang JingtangUnited Iron and Steel Co. Ltd. Tangshan 063210， Hebei， China）

Abstract： During the smelting process of high-grade pipeline steel X80， the control of large inclusions is one of the key factors to ensure its performance stability and engineering applicability. Through intensive sampling and system ⁃ atic analysis in the LF（ladle furnace）-VD（vacuum degasser） refining process， it was found that large inclusions of pipeline steel X80 concentrated after VD calcium treatment， and the main types were C12A7（12CaO·7Al2O3） and CA2（CaO·3Al2O3）. In the soft blowing stage， the large inclusions were not effectively removed， and the number of large inclusions instead showed an increasing trend. Therefore， it is judged that the current VD soft blowing process control is unreasonable， resulting in the aggregation and growth of inclusions due to insufficient stirring dynamics to form large inclusions. Combined with the parameters of ladle equipment and process control standards， the threedimensional multi-physical field numerical simulation of the VD soft blowing stage of pipeline steel was carried out. A comprehensive analysis was carried out from aspects such as flow field distribution， inclusion migration paths， and the stirring intensity of molten steel. It is pointed out that the original soft blowing process cannot form an effective inclusion migration channel under the current metallurgical reaction conditions. Combined with the simulation results， a new VD soft blowing control scheme was proposed. The soft blowing flow rate was increased from 50 L/ min to 60 L/min. While ensuring the uniformity of the molten steel temperature， the internal convection intensity of the molten steel and the floating efficiency of inclusions were significantly enhanced. After implementing the opti⁃ mized soft blowing process， verified by industrial tests， the large inclusions number density in the molten steel at the end of soft blowing decreased from 0. 074/mm2 to 0. 020/mm2， and the large inclusions number density in the pipe⁃ line steel decreased from 0. 071/mm2 to 0. 040/mm2， significantly improving the quality of the molten steel and the qualified rate of flaw detection for pipeline steel. This research achievement provides an effective technical path for achieving fine control of inclusions in the LF-VD refining process of high-grade pipeline steel X80， and also offers a reference for the subsequent optimization of clean smelting processes for other high-end steel grades.

Key words： pipeline steel； high-level； large inclusion； LF-VD refining process； soft blowing flow rate；numericalsimulation； industrial test； flaw detection

近年来，管线钢服役环境逐步向大口径以及耐高压、高冷、较强腐蚀方向发展，对管线钢冶炼过程钢液质量和夹杂物控制要求更为严格［1-4］。非金属夹杂物在成分、硬度和热膨胀性等方面区别于钢体，很容易造成管线钢氢、硫致开裂及焊接性能恶化等问题［5-6］。目前，冶炼高级别管线钢的精炼工艺流程主要有LF+RH（循环真空脱气炉）和LF+VD 这2 种，国内外专家学者对LF+RH 工艺进行了大量研究［7-10］，但LF+VD 工艺冶炼管线钢的研究成果鲜见报道。

板坯探伤是检查板坯夹杂缺陷的主要手段，近年来，高级别管线钢探伤结果中的大型夹杂物问题频发，关于LF+VD 工艺下大型夹杂物的形成机制，大部分学者认为是由于顶渣控制不合理导致的［11-13］，当精炼渣碱度过大时，钢中容易出现大型类夹杂物，需要控制合适的顶渣碱度和钙铝比。部分学者则认为大型夹杂物主要在VD 真空处理过程形成［14-16］，是强烈的钢渣反应在高真空条件下导致钢中增钙的结果。还有部分观点认为大型夹杂物是由于卷渣导致的［17］。在工业生产中，往往存在顶渣成分调整波动大、真空度参数无法变更及卷渣随机性等问题。

软吹作为精炼过程夹杂物上浮去除的最后1 个流程，可以有效地解决大型夹杂物问题，而且VD 软吹参数调整不直接影响工业生产节奏，实用性强。因此，有必要从该角度进行研究，通过获得最优VD软吹工艺参数来达到去除高级别管线钢大型夹杂物的目的。

# 1　管线钢 X80 工业生产流程及取样方法

# 1. 1 工业生产流程

以首钢京唐高级别管线钢X80 为研究对象，冶炼流程为 BOF（basic oxigen furnace）→LF→VD→CC（continuous caster），在精炼工序进行多级取样、系统分析，管线钢 X80 冶炼流程如图 1 所示。顶底复吹转炉冶炼全程底吹氩，采用前后挡渣出钢，出钢过程中进行炉后脱氧合金化；LF 过程前期快速形成白渣，终渣 w（MnO+FeO）≤1%，VD 深真空时长 20 min；破空后通过喂丝机向钢液内加入钙线，进行钙处理，保证铸机可浇性；钙处理后进行软吹，混匀钢液成分和温度，促使夹杂物上浮，最后上连铸机浇注成板坯后送轧机进行轧制。

![](images/7c2ea98e67cf42dd2a22716f9b0dc50e107cde6549e71f57778d36c190b4ee25.jpg)  
图 1 管线钢X80 冶炼流程  
Fig. 1 X80 pipeline steel smelting process

# 1. 2 取样位点及方法

在LF 进站、升温、出站、VD 进站、破空、钙处理及软吹等环节取钢液、顶渣及提桶样，采用光直谱分析仪检测管线钢X80 化学成分，结果见表1。

表 1 X80 化学成分（质量分数）  
Table 1 X80 chemical composition（ mass fraction）  
  

<html><body><table><tr><td></td><td>Si</td><td>Mn</td><td>D</td><td></td><td>Alt</td><td>Cu</td><td>Ni</td><td>Cr</td><td>Mo</td><td></td><td>Ti</td></tr><tr><td>0.0470</td><td>0.2000</td><td>1.7200</td><td>0.0070</td><td>0.0016</td><td>0.0310</td><td>0.1400</td><td>0.1600</td><td>0.1400</td><td>0.0700</td><td>0.0100</td><td>0.0140</td></tr></table></body></html>

通过现场钢液取样器进行钢样采集，使用氧氮氢分析仪分析钢液中氧氮成分变化。在不同生产阶段，使用烧氧管沾取钢液顶渣，冷却后敲碎，使用高清相机拍照观察顶渣形貌和颜色变化。分阶段挑 选 尺 寸 合 适 的 代 表 渣 进 行 X 射 线 荧 光 光 谱（XRF）成分分析。对提桶样进行制样（尺寸为30 mm×30 mm×30 mm）、打磨和抛光后，使用扫描电镜仪器分析夹杂物成分组成，使用 ASPEX 夹杂物分析仪分析夹杂物形貌和组成。

# 2 取样结果及分析

# 2. 1 钢液中氧、氮分析

不同精炼阶段钢液氧、氮含量变化如图2 所示，在 LF 开始前，钢液中氧质量分数为 0. 004 3%，而转炉终点氧质量分数为 0. 045% 左右，这说明出钢脱氧合金化效果明显。在 LF 精炼期间，由于加入的钢、渣铝质脱氧剂及钢液中的硅持续发生脱氧反应，氧质量分数由 0. 004 3% 降低至 0. 002 1%。钢液经过 VD 底吹搅拌和真空处理，氧质量分数降低至 0. 000 9%，钙 处 理 后 ，氧 质 量 分 数 增 加 至0. 001 1%。因为钙处理期间反应剧烈，造成了钢液吸氧，软吹过程钢液氧含量变化平缓，氧质量分数为 0. 001 2%。钢液中氮质量分数在 LF 处理期间呈 上 升 趋 势 ，由 进 站 的 0. 004 5% 增 加 至0. 006 7%，说明在LF 进行升温和深脱硫期间，钢液发生了吸氮现象。经过 VD 处理后，氮质量分数降低至0. 003 2%，VD 脱氮率超过50%，钙处理后，氮质量分数上升至0. 003 5%，软吹期间保持不变，这与氧含量变化趋势一致。

![](images/408707ee7fe52b265fbcf371c89940d434e163a0f3bd275e6b86d5549c6bdcff.jpg)

# 2. 2 顶渣分析

样品顶渣形貌和颜色变化如图3 所示，LF 进站后，通过快速造白渣，顶渣颜色变化明显，二次升温较一次升温颜色偏暗，VD 真空过程顶渣颜色无明显变化，钙处理后颜色变浅，之后无变化。

顶渣的成分见表2，为便于统计和计算，本文将w（CaO）与 w（SiO2）比值定义为二元碱度 R，将w（CaO）与 w（Al2O3）比值定义为钙铝比 C/A。LF进站后，SiO 质量分数处于高位，为 8. 06%；造渣完 成 后 ，w（CaO）为 50%\~55% ，w（Al2O3）为34%\~38%，2 种成分质量分数之和超过80%，其他各成分质量分数不超过 10%。代表顶渣氧化性的FeO 和 MnO，质量分数之和在 1% 以下，说明顶渣氧化性控制较好。

![](images/107b9dcb8b837a947beba3d8f515c6395c28331547aa02711140f5d8a3770c45.jpg)  
图 2 钢液中w（T［O］）和w（T［N］）变化 Fig. 2 Changes of w（T［O］）and w（T［N］） in molten steel   
图 3 顶渣形貌和颜色变化  
Fig. 3 Morphology and color change of top slag

表 2 LF 和VD 过程渣成分构成  
Table 2 Composition of slag during LF and VD   

<html><body><table><tr><td>Process</td><td>w(CaO)/%</td><td>w(SiO2)/%</td><td>w(MgO)/%</td><td>w(A12O3)/%</td><td>w(MnO)/%</td><td>w(FeO)/%</td><td>C/A</td><td>R</td></tr><tr><td>LF arrival</td><td>44.00</td><td>8.06</td><td>4.44</td><td>30.84</td><td>7.12</td><td>1.19</td><td>1.43</td><td>5.46</td></tr><tr><td>The first heating</td><td>55.50</td><td>2.84</td><td>5.26</td><td>34.20</td><td>0.09</td><td>0.17</td><td>1.62</td><td>19.55</td></tr><tr><td>The second heating</td><td>53.12</td><td>2.63</td><td>5.68</td><td>34.98</td><td>0.23</td><td>1.86</td><td>1.52</td><td>20.21</td></tr><tr><td>LF ending</td><td>54.03</td><td>3.04</td><td>5.36</td><td>34.98</td><td>0.13</td><td>0.56</td><td>1.54</td><td>17.75</td></tr><tr><td>VD arrival</td><td>52.04</td><td>3.27</td><td>6.61</td><td>36.56</td><td>0.21</td><td>0.65</td><td>1.42</td><td>15.93</td></tr><tr><td>Vacuum ending</td><td>50.76</td><td>2.85</td><td>6.62</td><td>38.23</td><td>0.08</td><td>0.10</td><td>1.32</td><td>17.79</td></tr><tr><td>VD ending</td><td>50.72</td><td>2.60</td><td>6.73</td><td>37.80</td><td>0.10</td><td>0.41</td><td>1.34</td><td>19.50</td></tr></table></body></html>

二元碱度与钙铝比变化趋势如图4 所示。R 值在LF 进站处于最低位，一次升温后升至19. 55，LF结束后有所降低，经过 VD 工序后又上升至高位。一次升温后，CaO 质量分数高达 55%，此后无明显变化，原因为加入了大量的石灰进行造渣脱硫。而SiO 降低与 Al O 升高呈明显的对应关系，可推断钢液中的铝还原了部分的SiO2 。LF 精炼过程C/A值为 1. 5\~1. 6，VD 期间降低 0. 1 左右，与这期间w（Al O ）升高而w（CaO）降低有关［18］。

![](images/c838d8e78d1a0e3f7851947694b58cc526932766aa45170f8d481a3e111f8b4c.jpg)  
图 4 顶渣 R 和 C/A 变化

# 2. 3 精炼过程夹杂物转变分析

通过对 LF 进站至 VD 软吹结束共 8 个过程进行取样和ASPEX 分析，检测不同尺寸夹杂物（本文定义直径不小于 10 μm 为大型夹杂物，小于 10 μm为小型夹杂物）在各个过程的变化情况，夹杂物扫描直径大于 1 μm，扫描面积大于 50 mm2，结果如图5 所示。LF 进站时，主要尺寸为（1，10） μm 的小型夹杂物，几乎没有大型夹杂物；一次升温后，尺寸为［10，20］ μm 的大型夹杂物开始出现；LF 结束时，尺寸为［10，20］ μm 的夹杂物比例为10% 左右，VD 过程夹杂物尺寸没有明显变化；钙处理结束后，出现尺寸大于20 μm 的大型夹杂物，且此时尺寸为［10，20］ μm 的夹杂物比例达到 20%。这与钙处理工艺特点有关，外部原因为钙处理过程钢液部分裸露，发生二次氧化，内部原因为钙与铝、镁氧化物反应生成大型夹杂物。软吹结束后，大型夹杂物比例明显升高，小型夹杂物比例明显减少，其原因为，1）一部分小型夹杂物在软吹过程聚集长大形成大型夹杂物，降低小型夹杂物数量的同时，增加了大型夹杂物的数量；2）软吹过程大型夹杂物动态变化，小型夹杂聚集形成的数量大于被顶渣吸附去除的数量，这说明软吹过程去除大型夹杂物的效果较差。

![](images/a04adcc95a7987019257c59edf321256087cab76927c1abbac0742f8b2082e2c.jpg)  
图 5 精炼过程夹杂物直径变化  
Fig. 5 Change of inclusion diameter in refining process

精炼过程夹杂物平均成分变化如图 6 所示，出钢过程进行脱氧合金化，脱氧剂加入铝粒，转炉终点氧质量分数为0. 03%\~0. 05%，铝与钢液中的氧反应生成大量氧化铝夹杂，所以 LF 进站夹杂物主要成分为Al O 。LF 处理过程中，顶渣、耐火材料和钢液三者之间持续接触反应，钙、镁元素进入钢液，导致LF 处理结束后，夹杂物的MgO、CaO 的质量分数升高，夹杂物由 Al2O3 向 Al2O3-CaO-MgO 转变。VD 过程由于钢、渣剧烈翻腾，钢、渣与耐材冲刷加剧，Al2O3 被消耗，顶渣和钢包内衬中的 CaO、MgO进入钢液，夹杂物的MgO、CaO 的质量分数持续升高。钙处理结束后，CaO 质量分数成倍增加，Al2O3质量分数成倍减少，MgO 和CaS 质量分数分别呈降低和升高趋势。软吹结束后，Al2O3和MgO 质量分数略微降低，CaS 质量分数略微上升，CaO 质量分数变化不明显。

![](images/1f1c969e563fc3113f82b22b3d22a4055fd87cc7ed34b8ead96730194071c8a2.jpg)  
Fig. 4 Change of top slag R and C/A   
图 6 精炼过程夹杂物平均成分变化Fig. 6 Average composition change ofinclusions during refining

# 2. 4 板坯中夹杂物分析

夹杂物的控制是否有效主要通过板坯探伤结果验证，实际探伤发现，距板坯下表面1/5 处存在长为308 μm 的夹杂缺陷，缺陷的电镜分析结果如图7所示，从夹杂物成分构成来看，初步判断为钙铝酸盐类夹杂物。

![](images/91dd306d7db7659ec5920a6c3f78630876c1af3da62777afb01b5f58f818852a.jpg)  
Fig. 7 Composition of steel plate defect

对钢板缺陷处夹杂物做 ASPEX 扫描分析，结果如图 8 所示，从成分组成上看，主要为含 MgO 的低熔点CaO-Al O ，从夹杂物直径上看，多处存在大型夹杂物，位于液相区。对大型夹杂物做电镜分析，结果如图9 所示。分析夹杂物成分组成，判断夹杂物类型主要为C12A7和CA2。

![](images/a7f5c08f45bcf567d56a6a8bcc39a8ef6acf4fdacaa4138c337cc1cf3734a11e.jpg)  
图 7 钢板缺陷成分组成  
图 8 钢板夹杂物成分分析

![](images/cab41c2809fc83baceb229d10b4658f4d8530f04b2e7bed16c6d6c5363237efc.jpg)  
Fig. 8 Inclusion composition analysis of steel plate   
图 9 大型夹杂物成分组成  
Fig. 9 Composition of large inclusions

综合夹杂物在精炼过程的尺寸和成分变化特点可以发现，高级别管线钢X80 大型夹杂物在钙处理后出现，软吹过程中呈现增多趋势，这与通过软吹促使夹杂物上浮去除的工艺效果相悖［19-21］，因此，初步判断软吹工艺不合理，后续的研究方向为软吹工艺优化。

流体密度ρ 为常数，式（1）可简化为：

式中：ρ 为流体密度，kg/m3；t 为时间，s；u、v、w 分别为x、y、z 方向的速度分量，m/s。

3. 2. 2 动量守恒方程

# 3 VD 软吹工艺优化

# 3. 1 模型参数及基本假设

使用 Fluent 软件模拟首钢京唐 210 t 钢包软吹过程。由钢包参数绘制的钢包几何模型和网络模型如图 10 所示，共计 1 359 277 个网格。首钢京唐VD 炉软吹工艺参数包括软吹时间和软吹流量，其中，高级别管线钢要求软吹时间大于 12 min，上限为15 min（生产节奏限制），从连铸可浇性和钢板夹杂物反馈来看，软吹时间为 15 min 效果最好，所以当前高级别管线钢软吹时间按上限控制。软吹流量为 30\~70 L/min，现场实际控制在 50 L/min。本文从生产实际出发，选取 30、40、50、60 和 70 L/min 5 个流量进行模拟试验，流量选取在现场软吹工艺要求范围内，软吹时间与生产实际保持一致。

为更好地开展模拟工作，做如下假设：1）流体不可压缩，流体的密度和黏度不发生变化；2）不考虑两相之间的化学反应和能量交换；3）底吹氩气流量和压力恒定，氩气泡大小均匀 ［22-23］。

![](images/ee6dabd8ab8abf726c2a070cf091c137e2cff31b2bf1d74f00a1d6d7114f433c.jpg)  
图 10 钢包结构示意  
Fig. 10 Schematic diagram of ladle structure

# 3. 2 控制方程及边界条件

控制方程主要包括连续性方程，动量、能量守恒方程及湍流模型，方程见式（1）\~式（6）。

3. 2. 1 连续性方程

式中：p 为单位体积上所受的压力，N；ρg 为单位体积上所受的质量力，N；U 为流体的速度矢量；μ 为动力黏度，Pa·s。

3. 2. 3 能量守恒方程

式 中 ：∂(ρT /∂t 为 单 位 时 间 内 流 体 焓 的 变 化 ；div( ρuT) 为 流 体 流 动 导 致 的 焓 输 运 ，W/m3；div (( b/cp ) grad T ) 为 傅 里 叶 热 传 导 ；cp 为 定 压 比 热容，J/（kg·K）；T 为温度，K；b 为流体的传热系数，W/（m2·K）；Q source 为黏性耗散项。

# 3. 2. 4 湍流模型

标准模型k-ε 由湍动能k 方程和湍流耗散率ε 方程确定。

G k 是由于平均速度梯度引起的湍动能产生项。

式中：μ1 为湍流黏度，Pa·s；ui、uj 为流体的速度矢量，i，j 为方向；x i、x j 为流体的方向；G b 为由于浮力引起的湍动能k 的产生项，对不可压流体G =0；Y M为可压湍流中脉动扩张的贡献，不可压缩流体Y M=0；C 1ε、C 2ε、C 3ε 为经验常数；σk、σε 分别为湍动能 k 和耗 散 率 ε 对 应 的 Prandtl 数 ；Sk、Sε 为 用 户 自 定 义源项。

经验常数和其他定义参数见表3。

# 表 3 经验常数和其他定义参数

Table 3 Empirical constants and other defined parameters   

<html><body><table><tr><td>C1</td><td></td><td>3</td><td></td><td></td><td>G</td><td>YM</td><td>Sk</td><td>Se</td></tr><tr><td>1.44</td><td>1.92</td><td>0.09</td><td>1.0</td><td>1.3</td><td>0</td><td></td><td></td><td></td></tr></table></body></html>

# 3. 3 数值模拟结果与讨论

从中心纵截面模拟结果来看，流速区域主要分为低流速区、中流速区和高流速区，低流速区域平均流速小于 0. 015 L/min，中流速区域平均流速为 0. 015\~0. 030 L/min，高 流 速 区 域 平 均 流 速 大于 0. 030 L/min，如图 11 所示。当底吹氩流量为30 L/min 时，中部低流速区域所占比例较大，平均流速为 0. 015 m/s，不利于钢液搅拌。当底吹氩流量为 40 L/min 时，中部低流速区域减小，但平均流速依然较低。当底吹氩流量为 50 L/min 时，中部低流速区域达到最小，平均流速为 0. 026 m/s。当底吹氩流量达到 60 L/min 时，吹氩搅动的钢液面积明显增大，平均流速达到 0. 031 m/s。当底吹氩流量为 70 L/min 时，搅动面积和平均流速变化不明显，中部又出现低流速区，不利于钢液混匀和搅拌。

(a) (b) (c) (d) (e) Velocity/(m·s−1) 0.050 NAA 0.045 0.040 0.035 0.030 0.025 0.020 0.015 0.010 0.005 i: （a）30 L/min； （b） 40 L/min； （c）50 L/min； （d） 60 L/min； （e） 70 L/min

从渣-金界面模拟结果来看（图12），当底吹氩流量为 30 L/min 时，水平界面平均速度为 0. 005 m/s，界面几乎未发生扰动。当底吹氩流量由 40 L/min增加到 50 L/min 时，界面平均流速逐步升高，开始显露“ 泉眼”，但搅动面积不够大。当底吹氩流量达到 60 L/min 时，“ 泉眼”清晰可见，搅动面积明显增加，水平界面平均速度增加至 0. 015 m/s。当底吹氩流量为 70 L/min 时，水平界面平均速度没有明显变化，但在钢包边部出现流速区，此区域与钢包耐材直接接触，容易造成钢包内壁冲刷，引起钢包侵蚀，侵蚀物到钢液中容易形成大型外来夹杂物。

Velocity/(m·s−1)0 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 0.045 0.050（a） 30 L/min； （b） 40 L/min； （c） 50 L/min； （d） 60 L/min； （e）70 L/min

在钢包底吹模拟过程中，通过在钢液面中心处加入一定量的饱和 NaCl 溶液，利用数据采集系统检测钢包内电导率的变化情况，当电极的电导率变化值不超过其稳定值的±5% 时则认定达到稳定电导值，以达到稳定电导值时间确定为混匀时间。如图 13 所示，可以看出，当吹氩流量为 30 L/min 时，钢液的混匀时间最长，为 516 s；随着吹氩流量的增加，混匀时间递减，当吹氩流量为60 L/min 时，混匀时间最短，为 351 s；当吹氩流量为 70 L/min 时，混匀时间呈上升趋势。

相关研究表明［24-28］，增加软吹时间对夹杂物去除有明显改善效果。210 t 钢液在 50 L/min 软吹流量下混匀后，持续大约7 min；在60 L/min 的软吹流量下混匀，持续时间约 8 min，持续时间相差不大。由图 11 和图 12 模拟结果可知，60 L/min 软吹流量搅动面积大，钢液面未裸漏，钢包壁未冲刷，基本可以排除软吹工艺优化前后持续时间因素的干扰。

![](images/44fae9f717db915af19ede9baa60033effa3106dd8dcadaadce00ab3ac1ab876.jpg)  
图 13 不同流量下混匀时间对比Fig. 13 Comparison of mixing time underdifferent flow rates

综上所述，在 50 L/min 的软吹流量下，钢液搅动面积不大，混匀时间较长，不利于夹杂物上浮去除，当软吹流量为 60 L/min 时，钢液平均流速、搅动面积及混匀时间均达到最优，当软吹流量高于60 L/min 后，不仅混匀时间增加，而且钢包壁位置出现低速区，会造成钢包侵蚀，增加夹杂物发生概率，因此，最佳软吹流量为60 L/min。

# 3. 4 夹杂物转变的工业生产验证

由3. 1节可知，VD 软吹工艺优化前，现场软吹流量按 50 L/min 控制，优化后，软吹流量由 50 L/min提升至60 L/min。对工艺优化前后的VD 软吹阶段进 行 密 集 取 样 和 ASPEX 分 析 ，扫 描 面 积 为49\~51 mm2，原工艺条件下夹杂物转变情况如图14所示（圆圈大小代表夹杂物尺寸），当软吹3 min 时，大型夹杂物被顶渣大量吸附，数量大幅度减少；当软吹7 min 后，大型夹杂物成倍减少，但小型夹杂物增加；当软吹11 min 时，出现较大型的夹杂物，并且随着软吹的时间延长，没有被顶渣吸附掉，最终在钢板上形成夹杂，造成管线钢探伤不合格。相关研究表明［29-30］，小型夹杂物大部分上浮去除，小部分由于碰撞聚合形成新的大型夹杂物。在此软吹工艺下，最终软吹结束和钢板中大型夹杂物所占比例较高。

![](images/4c317cb27abff55d0f52e03227b9274427978e5b89c31b2a0f3f815b8d12e1f5.jpg)  
图 14 原工艺条件下夹杂物转变  
Fig. 14 Inclusion transformation under original process conditions

优化工艺条件下夹杂物转变情况如图15 所示，软吹前期，大型夹杂物首先被顶渣吸附，而后又逐步增加；软吹中期，大型夹杂物呈下降趋势，同时，小型夹杂物也相应减少；软吹后期，夹杂物整体数量均较中期大幅减少；软吹结束后，钢液中基本没有检测到大型夹杂物，说明优化后的软吹流量对夹杂物的去除效果较好。板坯中夹杂物尺寸和数量有上升趋势，这与连铸过程二次氧化有关。

![](images/932006d449e10e602e544833d3f1ce08e2a62e8a028bf27bc876192ce8440e70.jpg)  
（a） 钙处理结束； （b） 软吹 3 min； （c） 软吹 7 min； （d） 软吹 11 min； （e） 软吹 15 min； （f） 钢板  
图 15 优化工艺条件下夹杂物转变

优化前后的大型夹杂物密度对比如图16 所示，2 种工艺中，大型夹杂物均是在软吹过程减少，从降幅和最终结果看，优化后工艺明显优于优化前，软吹结束后钢液中大型夹杂物数量密度由0. 074 个/mm2降低至0. 020 个/mm2，板坯中大型夹杂物数量密度由 0. 071 个/mm2降低至 0. 040 个/mm2。

![](images/39ca85496dbea1d32c564e5984d34640d7fe15fcb75a76026830a499171e4492.jpg)  
Fig. 15 Inclusion transformation under optimized process conditions   
图 16 优化前后的大型夹杂物数量密度对比 Fig. 16 Comparison of large inclusion number density before and after optimization

# 4　结论

1）通过对原工艺精炼工序和板坯进行密集取样和分析发现，高级别管线钢 X80 在 VD 钙处理后出现了大量尺寸大于10 μm 的大型夹杂物，电镜分析夹杂物类型主要为钙铝酸盐，成分结构为 C A和CA2。该类夹杂物最终在板坯内部出现，造成板坯探伤不合缺陷。

2）管线钢X80 大型夹杂物数量在VD 软吹过程中没有减少，判断软吹工艺与夹杂物去除不匹配。通过对 VD 软吹不同流量区间进行流场模拟，发现在原工艺50 L/min 的软吹流量下，钢液搅动面积不大，混匀时间较长，夹杂物去除效果不佳；当软吹流量由 50 L/min 增加至 60 L/min 后，钢液平均流速、搅动面积及混匀时间均达到最优。

3）通过工业试验验证，采用软吹流量为60 L/min的新工艺后，软吹结束钢液中大型夹杂物数量密度由0. 074 个/mm2 降低至 0. 020 个/mm2，板坯中大型夹杂物数量密度由0. 071个/mm2降低至0. 040个/mm2，大型夹杂物去除效率得到明显改善。

# 参考文献：

［1］ 周桂娟，童志，陈晓华，等.  X80 管线钢焊接与焊缝开裂影响因素研究进展［J］.  材料导报，2022，36（2）：164.（ZHOU G J，TONG Z，CHEN X H，et al.  A review on the welding of X80pipeline steel and factors affecting weld cracking［J］. M ateri⁃als Reports，2022，36（2）：164. ）  
［2］ 刘阳，李学达，刘峻峰，等.  我国长输天然气用管线钢的发展现状与趋势［J］.  材料热处理学报，2024（3）：45.（LIU Y，LIX D，LIU J F，et al.  Development status and trends of pipe⁃line steel for long-distance natural gas transmission in China［J］. Journal of Materials and Heat Treatment， 2024（3）：45.  
［3］ 牛延龙， 刘清友， 贾书君， 等. 控冷工艺下组织及M/A 岛对管线钢韧性的影响［ J］. 钢铁， 2020， 55（6）： 91. （NIU Y L，LIU Q Y ，JIA S J，et al.  Influence of microstructure and M/A island evolution on toughness of pipeline steel under con⁃trolled cooling process［J］. Iron and Steel ，2020， 55（6）： 91.  
［4］ 周禹， 巨银军， 王嵘坤， 等 . 氩气泡致管线钢内部缺陷的控制理论与实践［J］. 钢铁 ，2024，59（10）： 45. （ZHOU Y ，JUY J， WANG R K， et al. Control theory and practice of argonbubble- induced internal defects in pipeline steel［J］. Iron andSteel，2024， 59（10）： 45. ）  
［5］ 姚婵，陈健，明洪亮，等. 管线钢氢渗透行为的研究进展［J］.中国腐蚀与防护学报，2023，43（2）：209. （YAO C ，CHEN J，MING H L，et al. Research progress on hydrogen permeabil⁃ity behavior of pipeline steel［J］. Journal of Chinese Societyfor Corrosion and Protection，2023，43（2）：209.  
6］ 刘祎， 董福涛， 齐程伟， 等 . 管线钢氢脆的研究进展［J］. 中国冶金， 2024， 34（7）： 11. （LIU Y， DONG F T， QI C W，al. Progress of hydrogen embrittlement in pipeline steel［J］. China Metallurgy，2024，34（7）：11.7］ 苑 波， 杨利彬， 赵进宣， 等 . 20CrMnTiH 齿轮钢在 BOF-LF-RH-CC 流程中的夹杂物演变［J］. 中 国 冶 金 ，2024，34（1）： 36. （YUAN Y B， YANG L B， ZHAO X， et al.Inclusions evolution in 20CrMnTiH gear steel during BOF-LF-RH-CC process［J］. China Metallurgy， 2024， 34（1）：36.  
［8］ 袁天祥， 张丙龙， 刘延强， 等 . 高级别管线钢夹杂物控制研究［J］. 中 国 冶 2020， 30（11） 85. （YUAN T X，ZHANG B L， LIU Y Q，et al.  Study on inclusion control ofhigh grade pipeline steel［J］. China Metallurgy， 2020， 30（11）：85. ）  
9］ 王康豪， 姜敏， 李凯轮， 等 . GCr15 轴承钢 BOF-LF-RH-CC流程夹杂物的生 成 及演 变［J］. 钢铁， 2022， 57（10）： 64.（WANG K H， JIANG M， LI K L， et al. Formation and evolution of inclusions GCr15 bearing steel produced by pro⁃cess of BOF LF -RH-CC［J］. Iron and Steel，2022， 57（10）：64. ）  
［10］ 钟华军，姜敏，王章印，等. X80 管线钢精炼过程夹杂物形成与演变［J］. 工程科学学报，2023，45（1）：98. （ZHONG H J，JIANG M， WANG Z Y，et al. Formation and evolution ofinclusions in the refining process of X80 pipeline steel［J］.Chinese Journal of Engineering，2023，45（1）：98.  
［11］ 付鹏冲，李文双，朱林林. 超低氧含量GCr15 轴承钢夹杂物控制［J］. 山东冶金，2015（6）：23. （FU P C，LI W S，ZHU L L.Study on inclusion control for ultra low oxygen GCr15 bear⁃ing steel［J］ Shandong Metallurgy，2015（6）：23.  
［12］ WANG C ，LIU Q，ZHANG S，et al. Optimization of VDrefining slag and control of non-metallic inclusions for55SiCrA spring steel［ C］//12th international symposium onhigh-temperature metallurgical processing. Cham： SpringerInternational Publishing， 2022：445.  
［13］ JIANG M，LI K L，WANG R G，et al. Cleanliness and con⁃trol of inclusions in Al-deoxidized bearing steel refined bybasic slags during LF-VD-Ar bubbling［J］. ISIJ Interna⁃tional，2022，62（1）：124.  
［14］ 徐迎铁，陈兆平， 杨宝权 . 轴承钢 DS 类大颗粒夹杂物研究［J］. 炼钢，2016，32（4）：49.（XU Y T，CHEN Z P，YANG BQ. Study of large size DS type inclusions in bearing steel［J］.Steelmaking，2016，32（4）：49. ）  
［15］ CAPURRO C，CERRUTTI G，CICUTTI C. Influence ofvacuum degassing on steel cleanliness［C］//9th InternationalConference on Clean Steel. Budapest： University of Mis⁃kolc，2015：7845.  
［16］ RIYAHIMALAYERI K，ÖLUND P. Development of oxideinclusions during vacuum degassing process［J］. Ironmakingand Steelmaking，2013，40（4）：290.  
［17］ 杨光维，陈兆平，柳向椿，等 .  VD 镇静时间对齿轮钢大颗粒夹杂物的影响［J］. 炼 钢 ，2019，35（6）：31. （YANG G W，CHEN Z P，LIU X C，et al.  Effect of holding time after VDon large-sized inclusions of gear steel［J］. Steelmaking，2019，35（6）：31. ）  
［18］ 刘延强，张鹏，何文远，等. X65 管线钢全流程夹杂物转变规律及控制［J］. 钢 铁 ，2018，53（12）：44. （LIU Y Q，ZHANGP，HE W Y，et al. Transformation behavior and control ofinclusions in X65 pipelinesteel during whole process［J］. Ironand Steel，2018，53（12）：44.  
［19］ 邢佳， 杜晓建，张欣杰，等. 喂钙量与软吹氩对316L 不锈钢中夹杂物的影响［J］ 特殊钢，2022，43（2）：31. （XING J，DU XJ，ZHANG X J，et al. Effect of calcium addition amount andsoft argon blowing on inclusions in 316L stainless steel［J］.Special Steel，2022，43（2）：31.  
［20］ 孟耀青， 李建立，朱航宇， 等 . LF 软吹时间对硅脱氧弹簧钢氧化物夹杂控制影响［J］. 钢铁，2022，57（5）：48. （MENG Y Q，LI J L， ZHU H Y， et al. Effect of LF soft bubbling time onoxide inclusions in Si-killed spring steel［J］.  Iron and Steel，2022，57（5）：48.  
［21］ 魏光升， 董建锋， 朱荣， 等 . 钢包底吹对RH 脱氢和夹杂物的影响［J］. 钢铁， 2021， 56（2）： 63. （WEI G S， DONG J F，ZHU R，et al. Effect of ladle bottom blowing on RH dehydro⁃genation and inclusions［J］. Iron and Steel，2021， 56（2）：63. ）  
［22］ 李子豪， 谢清华， 厉英， 等 . 电弧炉复合喷吹对熔池流动及混匀行为的影响［J］. 钢铁，2025，60（6）：77.（LI Z H， XIEQ H，LI Y ，et al. Effect of EAF co-injection on flow andmixing behavior of molten pool［J］.  Iron and Steel，2025，60（6）：77. ）  
［23］ 钱云强，郑淑国，朱苗勇. 偏心底吹氩钢锭流场及混合特性数值 模 拟［J］. 中 国 冶 金 ，2022，32（11）： 56. （QIAN Q，ZHENG S G，ZHU M Y. Numerical simulation of flow fieldand mixing characteristics in steel ingot with eccentric bottomblowing Argon［J］. China Metallurgy，2022，32（11）：56.  
［24］ CHENG G，ZHANG F，REN Y，et al. Evolution of nonme⁃tallic inclusions with varied argon stirring condition duringvacuum degassing refining of a b earing steel J］.  Steel ResearchInternational，2021， 92（1）：2000364. （下转第123 页）deformation in casting rolls［J］. International Communica⁃tions in Heat and Mass Transfer， 2025，164：108936.  
［25］ LI Y，HE C，LI J D，et al. A novel approach to improve themicrostructure and mechanical properties of Al-Mg-Si alumi⁃num alloys during twin-roll casting［J］. Materials， 2020， 13（7）：1713.  
［26］ 宋黎，孙斌煜，崔鹏鹏，等 . 基于 ANSYS 的镁合金铸轧辊冷却研究［J］. 有色金属材料与工程，2017，38（3）：144. （SONGL，SUN B Y，CUI P P，et al. Cooling research of magnesiumalloy casting-roll using ANSYS software［J］. Nonferrous MetalMaterials and Engineering， 2017，38（3）：144. ）  
［27］ 宋黎，孙斌煜，乔东洋.  不同冷却水流速度下的铸轧辊套温度分析［J］. 大型铸锻件，2017（3）：27. （SONG L，SUN B Y，QIAO D Y . Analysis on the temperature of casting rollsleeve under different cooling water velocity［J］. Heavy Cast⁃ing and Forging，2017（3）：27. ）  
［28］ 张彦荣，石怀鑫，魏振鹏，等.  铝合金双辊铸轧轧辊横向冷却强度分布［J］. 有色金属工程， 2023，13（3）：41. （ZHANG YR，SHI H X，WEI Z P，et al. Transverse cooling strength dis⁃tribution of aluminum alloy twin-roll casting rolls［J］. Nonfer⁃rous Metals Engineering，2023，13（3）：41. ）  
［29］ HAGA T，OKAMOTO T， WATARI H，et al. Effect of rolldiameter on temperature of aluminum alloy strip during cast⁃ing using a vertical type high speed twin roll caster［J］. Mate⁃rials Science Forum ，2024，116：43.  
［30］ 陈思阳 铸轧辊冷却水道设计及其对冷却效果的影响［D］.鞍山： 辽宁科技大学，2019. （CHEN S Y . Casting roll cool⁃ing channel design and its effect on cooling effect［D］. Anshan：Liaoning University of Science and Technology，2019. ）  
［31］ WANG Y C，ZHANG X M，ZHANG Y A，et al. The effectof the thermal deformation of casting roll on strip thickness inthe strip casting process ［J］. Steel Research International，2022， 93（11）：2200115.  
［32］ LAMARCHE-GAGNON M É， MOLAVI-ZARANDI M，RAYMOND V， et al. Additively manufactured conformalcooling channels through topology optimization［J］. Struc⁃tural and Multidisciplinary Optimization， 2024， 67（8）：138.  
［33］ CHENG G，YIN N，ZHENG Q，et al. Numerical simulationof surface thermal analysis and cooling optimization of con⁃tinuous casting rolls［J］. Crystals， 2024， 15（1）：41.  
［34］ 季策，黄华贵，赵千里，等.  一种具有铸轧辊分区控温功能的高 速 宽 幅 铸 轧 设 备 及 方 法 ：CN202410873849. 3［P］. 2024-07-02. （JI C，HUANG H G，ZHAO Q L，et al. High-speedwide-width casting and rolling equipment and method with atemperature control function for casting rolls：CN202410873849.3［P］.  2024-07-02. ）  
［35］ WANG W，CAO Y，OKAZE T. Comparison of hexahedral，tetrahedral and polyhedral cells for reproducing the wind fieldaround an isolated building by LES［J］. Building and Environ⁃ment， 2021， 195：107717.  
［36］ JI C，HUANG H G，SUN Y F，et al. Research on flow rateengineering calculation model of U-pipes for fabricating trans⁃verse variable profiled（TVP）strips by simulation and experi⁃ment［J］.  Metallurgical Research and Technology， 2020， 117（6）：604.  
［37］ SMITS A J，MCKEON B J，MARUSIC I. High-reynoldsnumber wall turbulence［J］. Annual Review of Fluid Mechan⁃ics，2011，43：353.

# （上接第91 页）

［25］ LIU C，GAO X，UEDA S，et al. Change in composition ofinclusions through the reaction between Al-killed steel andthe slag of CaO and MgO saturation［J］. ISIJ International，2019， 59（2）：268.  
［26］ 马超，刘毅，张福利，等. 软吹对GCr15 轴承钢洁净度的影响［J］. 炼钢，2024，40（3）：33. （MA C，LIU Y，ZHANG F L，etal. Effect of soft blowing on the purity of GCr15 bearing steel［J］. Steelmaking， 2024，40（3）：33. ）  
［27］ 刘风刚，任英，段豪剑，等.  钢包软吹过程优化数学模拟和工业试验研究［J］. 炼钢，2019，35（6）：24.（LIU F G，REN Y，DUAN H J，et al. Mathematical simulation and plant trial onsoft blowing process of ladle furnace［J］. Steelmaking， 2019，35（6）：24. ）  
［28］ RAVI G，DE WAELE W，NIKOLIC K，et al. Numericalmodelling of rolling contact fatigue damage initiation fromnon-metallic inclusions in bearing steel［J］. Tribology Interna⁃tional， 2023， 180：108290.  
［29］ 吴明晖，程礼梅，尹青，等.  钢/Ar 及钢/渣界面非金属夹杂物碰撞团聚行为原位观察［J］.  工程科学学报，2025，47（4）：727.（WU M H，CHENG L M，YIN Q，et al. In situ observa⁃tion of collision and agglomeration behavior of non-metallicinclusions at steel/Ar and steel/slag interfaces［J］. ChineseJournal of Engineering，2025，47（4）：727. ）  
［30］ 张静， 刘瀚泽， 张世锟， 等 .  第一性原理计算方法在钢中夹杂物研究中的应用［J］.  中国冶金， 2023， 33（8）：6.（HANGJ， LIU H Z， ZHANG S K， et al. Application of first-principles calculation method in study of inclusions in steel［J］. China Metallurgy， 2023， 33（8）：6. ）
