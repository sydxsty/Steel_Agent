# 基于局部应变能密度的厚板T 形接头焊趾和焊根疲劳性能分析

刘夕1， 陈广冉2， 孟庆禹1， 龚宝明2， 王灿1(1. 江苏徐工工程机械研究院有限公司，徐州，221004； 2. 天津大学，天津，300350)

摘要： 以装载机前车架疲劳寿命评估为目的，从前车架焊接结构关键部位中提取三种典型焊接接头，对其进行疲劳试验和分析.以局部应变能密度法为理论基础， 对获得的试验数据进行处理，得到了基于局部应变能密度的疲劳寿命曲线，并采用文献中十字接头的疲劳数据验 论的准确性和适用性.结果表明，局部应变能密度法既 用一焊根失效疲劳评估，也可以用于焊趾失效疲劳 估 不同位置的疲劳裂纹 会影响结构的刚度， 进而影响结寿命.此外，研究了焊脚尺寸和熔深对焊接接 劳寿命的影响，推导了基于局部应变能密度法的疲劳失效 置转换区间.并通过文献中的疲劳数据验证了文中提出的焊趾失效和焊根失效转换区间的准确性，为合理设计未焊透厚板T 形接头抗疲劳性能提供依据.

关键词： 疲劳评估；有限元；应变能密度

中图分类号：TG 405 文献标识码：A doi：10.12073/j.hjxb.20191112001

# 0 序言

工程机械领域的大型焊接结构件在工作过程中多承受循环载荷，容易于焊缝处产生疲劳开裂而导致结构件失效.据统计，在焊接结构的失效中，有70% \~ 90% 是由于焊接接头的疲劳断裂造成的[1-2].因此，研究准确的焊接结构疲劳寿命评估方法和接头抗疲劳设计具有重要的意义.

目前有三种以 S-N 曲线为基础的方法[3-4]，包括名义应力方法、热点应力方法和等效缺口应力方法.以上设计方法已经纳入国际焊接学会钢结构疲劳设计规范中[5].同时也纳入了一些国际行业设计规范，如挪威船级社的海洋钢结构疲劳设计规范 DNV-RP-C203[6].然而，由于这些方法自身的缺点大大限制其在大型复杂结构中的疲劳评估.名义应力方法虽然简单，但由于忽略了过多的结构细节，导致很多接头形式未被标准收纳，因此容易造成评估结果的较大偏差；热点应力法通过有限元计算或试验测量将结构性应力集中考虑在循环应力中，但是对于焊根疲劳失效及破坏厚度方向应力线性分布的假设的结构形式是不适用的；缺口应力法的优点是采用一条标准的设计曲线，可以针对几乎所有不同细节焊接接头进行疲劳设计，但是需要建立考虑构件所有焊接部位的焊趾、焊根几何形状与相对位置并且包含错边、角变形等细节的有限元模型，等效缺口应力法要求精细网格、单元数量过多、计算规模巨大，因而对设计人员和计算机软硬件配置均提出很高要求，对大型结构的设计是比较困难的[4].

为了改善传统焊接结构疲劳评定方法的局限性，Lazzarin 等人[7] 在断裂力学的基础上，结合能量法推导出了局部应变能密度法，并用其对大量焊接疲劳数据进行处理，证明了该方法在焊接结构疲劳评估领域的适用性和优越性.该方法不仅考虑了载荷模式、接头类型以及几何尺寸对疲劳寿命的影响，克服了传统疲劳评估方法存在的网格敏感性问题，而且能统一考虑焊趾失效和焊根失效及不同失效位置之间的转换.

有鉴于此，采用局部应变能密度法处理厚板T 形接头的疲劳数据，得到了基于局部应变能密度的疲劳寿命曲线，以及疲劳失效位置转换区间图，为焊接接头疲劳设计和失效位置控制提供了理论依据.

# 局部应变能密度法

局部应变能密度法是以缺口尖端定长半径扇形内的应变能密度为参量，来描述结构的疲劳行为.能量法认为，结构受到外载荷作用时由于发生变形，会在内部积累一定的能量，当这种能量达到一定值时，结构会发生破坏而失效.对于焊接结构，主要关心焊根和焊趾缺口尖端处扇形内的应变能密度，如图 1 所示.

Williams[8] 通过研究量化了 I 型和 II 型两种载荷模式下，由于缺口存在而造成的应力场的奇异性.如图 1 所示的焊接接头中，在焊根和焊趾缺口尖端处划定半径为 Rc 的扇形区域，当焊趾处的半径为 0 时，靠近缺口尖端的应力场的强度可以用应力强度因子来描述.通过在扇形区域内引入极坐标系 (r, θ)，I 型和 II 型两种情况下的应力强度因子为[9]

![](images/60b1ac74e4705a683955e767781c09b15e5201382fe7586a710e9442b7863432.jpg)  
图 1 焊根和焊趾处的几何参数  
Fig. 1 Geometric parameters at the root and weld toe

式中：σθθ和τrθ是焊根和焊趾缺口平分线上的应力分量， λi (i=1, 2, 3) 分别 为 I 型、 II 型 和 III 型载荷作用下的特征值.在 III 型载荷作用下，Zappa-lorto 等人[10] 扩展了应力强度因子的定义，其表达式为

进一步可以得到用于描述缺口尖端应力场的 Williams 公式.在 I 型模式下，缺口尖端的应力分布为[11]

式中：χ 为应力函数.在 II 型载荷作用下，缺口尖端 的应力分布为在 III 型载荷作用下，缺口尖端的应力分布为

证过[13-14].

在平面应变情况下，扇形内的平均应变能密度为[13-14]

式 (1) \~ 式 (7) 中的参数，只和缺口的张开角有关，表 1 给出了 I 型、II 型及 III 型三种载荷模式作用下的参数和缺口张开角的关系.

其中，

如上所述计算应变能密度时，最关键的步骤是临界半径 Rc 的确定，该参数只和材料有关. Livi-eri 等人[12] 通过对各种焊接接头疲劳数据进行处理，认为对钢材质的接头，取 Rc=0.28 mm 是最合理的.文中对应变能密度的计算采用应变能除以相应体积的方式，该方法的正确性已被大量试验数据验式中：E 为材料的弹性模量；Rc 为扇形半径；ΔKiN为不同载荷模式作用下的应力强度因子；ei 和 λi 是只和缺口张开角有关 2α 的参数，具体数值如表 1所示.

<html><body><table><tr><td rowspan="2">2α</td><td colspan="2">Mode I</td><td colspan="2">Mode II</td><td colspan="2">Mode III</td></tr><tr><td>λ1</td><td>1</td><td></td><td>e2</td><td>λ3</td><td>e3</td></tr><tr><td>0</td><td>0.500</td><td>0.133</td><td>0.500</td><td>0.340</td><td>0.500</td><td>0.413</td></tr><tr><td>π/6</td><td>0.501</td><td>0.147</td><td>0.598</td><td>0.274</td><td>0.545</td><td>0.379</td></tr><tr><td>π/3</td><td>0.512</td><td>0.151</td><td>0.731</td><td>0.217</td><td>0.600</td><td>0.344</td></tr><tr><td>π/2</td><td>0.544</td><td>0.145</td><td>0.909</td><td>0.168</td><td>0.666</td><td>0.310</td></tr><tr><td>2π/3</td><td>0.616</td><td>0.129</td><td>1.149</td><td>0.128</td><td>0.750</td><td>0.275</td></tr><tr><td>3π/4</td><td>0.674</td><td>0.118</td><td>1.302</td><td>0.111</td><td>0.800</td><td>0.258</td></tr></table></body></html>

# 疲劳试验及有限元模拟

前车架是装载机的重要组成部件，是连接工作装置和后车架的中间机构.前车架中的典型结构大多是由厚度为 8 \~ 30 mm 的 Q345 钢板以连续对接或角接的形式焊接而成.文中材料母材选用和前车架相同的 Q345 钢，焊丝选用 ER50-6，母材和焊丝的化学组成及力学性能如表 2 及表 3 所示.疲劳试件采用 TIG 焊接而成，焊机型号为 Fronius tps5000，焊接参数如表 4 所示.

对装载机前车架焊接结构进行整体分析后确定出其危险焊缝中含有的三种接头类型，如图 2 所示，分别为未焊满和满焊，而满焊接头又根据焊脚尺寸的大小分为满焊大尺寸和满焊小尺寸. 接头尺寸如图 2c 所示，主板长宽厚分别为 180，130 和16 mm，腹板长宽厚为 100，60，46 mm.

表 1 平面应变情况下各参数值 (泊松比 0.3)Table 1 Plane strain parameters under different conditions (Poisson ratio 0.3)  
表 3 母材及焊丝的力学性能  

<html><body><table><tr><td>材料</td><td>C</td><td>Mn</td><td>Si</td><td></td><td>S</td><td>√</td><td>Fe</td></tr><tr><td>Q345</td><td>0.2</td><td>1.0~ 1.6</td><td>0.55</td><td>0.045</td><td>0.045</td><td>0.02~0.15</td><td>余量</td></tr><tr><td>ER50-6</td><td>0.1</td><td>1.18</td><td>0.25</td><td>0.009</td><td>0.02</td><td></td><td>余量</td></tr></table></body></html>

Table 3 Mechanical properties of base metal and filler   

<html><body><table><tr><td>材料</td><td>屈服强度ReL/MPa</td><td>抗拉强度Rm/MPa</td><td>断后伸长率A(%)</td></tr><tr><td>Q345</td><td>345</td><td>460~630</td><td>21</td></tr><tr><td>ER50-6</td><td>330</td><td>490</td><td>30</td></tr></table></body></html>

表 2 母材和焊丝的化学组成(质量分数， %)Table 2 Chemical compositions of base metal and filler  
表 4 焊接工艺参数Table 4 Welding parameters  

<html><body><table><tr><td>焊接电压U/V</td><td>焊接电流I/A</td><td>焊接速度v/(mm·min−1)</td></tr><tr><td>24</td><td>200</td><td></td></tr></table></body></html>

![](images/0b2d49c4f3608743cd9528e0f4a6fdb6e37f0b98b392af14f2b192a3fe3e943e.jpg)  
图 2 接头类型及几何尺寸  
Fig. 2 Geometrical characteristic of joint. (a) lack of penetration; (b) full penetration; (c) dimensions of joints

采用 ANSYS 软件进行有限元仿真，为节省计算时间，建立二分之一模型，如图 3 所示.建模分为两步，第一步是根据焊缝处单个单元长度上的应变能密度进行网格敏感性分析，以确定焊缝方向上最合适的单元长度.设置焊缝长度的单个单元长度为2 \~ 10 mm 之间不等的数值，分别取焊缝端部单个单元和整条焊缝的应变能密度，分析其和单元长度之间的关系.模拟结果显示：随着单元长度的增加，整条焊缝和单个单元长度内的应变能密度均先减小再增加；在单元长度小于 5 mm 时，焊缝和单个单元内的应变能密度变化较小.为后续建模及计算方便，焊缝方向的单元长度设为 5 mm.采用Solid186 单元，材料属性设置弹性模量为 206 GPa，泊松比为 0.3.在焊根和焊趾处分别建立半径为0.28 mm 的扇形，扇形内最小网格尺寸为 1×10−3 mm.在焊根和焊趾较近的区域建立细网格，远离焊根和焊趾的地方逐渐过渡为粗网格.有限元模型包含366 596 个节点，172 700 个单元，如图 3 所示.

![](images/f94f12cfd2bf2fb80f75a55ad1d480ff8df60e7be0ea48fe78f1a4660f99b970.jpg)  
图 3 有限元网格Fig. 3 Finite mesh of T-joint

# 3    结果分析

# 3.1 应变能密度曲线分析

以局部应变能密度法为理论基础，对试验和模拟数据进行处理，得到了基于应变能密度的 S-N 曲线，如图4 所示.为了验证该S-N 曲线的准确性和适用性，文中选取了文献 [15] 中的部分疲劳数据进行拟合.可以看到，两者具有相同的斜率，但是文献[15] 的试样疲劳寿命略大于三种T 形接头.考虑到文献[15]试样均在焊趾处断裂，文中的 T 形接头均在焊根处断裂，可能是由于不同位置的疲劳裂纹导致结构的刚度不同，进而导致疲劳寿命的不同.

为验证上述猜想，进行了如下研究.采用如图 5所示的十字接头，分别在焊根和焊趾处预置不同长度的裂纹，施加轴向拉伸载荷，用载荷施加端的最大位移代替刚度，结果如图 6 所示.由图可以看出，裂纹长度一定时，在焊根处预置裂纹，其载荷施加端的最大位移大于在焊趾处预置裂纹时载荷施加端的最大位移.相应的，在焊根预置裂纹时其结构刚度小于在焊趾预置裂纹，进而疲劳寿命较小，与图 3 的结果相吻合.

![](images/4f73bd92d53258ba92b7101a4dba053eaf51c819ce80b9734a4679e2757e2f97.jpg)  
图 4 基于应变能密度的 S-N 曲线Fig. 4 S-N curve based on strain energy density

![](images/d5317733fc755d183410d8a3f424e7806fc4b4dc44503b1165877c06d14b473e.jpg)  
图 5 预置裂纹的十字接头Fig. 5 Pre-cracked cruciform joint

![](images/330fef33655105620e283dc5dd351725e1a811aef5038bd58cde23eebba42cbf.jpg)  
图 6 位移和裂纹位置的关系Fig. 6 Relationship between displacement and crackposition

# 3.2 疲劳失效位置转换区间分析

尺寸效应对疲劳寿命的影响不可忽视，并且Kihl 等人[16] 已经研究了板厚的影响.鉴于此，文中基于上述的结果，主要研究焊脚尺寸 (s) 和熔深(p) 对疲劳寿命的影响.依然采用图 5 所示的十字接头，设置板厚为 T = t = 10 mm，改变熔深和相对焊脚尺寸，分别取焊趾和焊根处的应变能密度，结果如图 7 所示.从图 7a 和 7b 中可以看出，焊脚尺寸固定时，其应变能密度均随着熔深的增加而减小；即应变能密度在焊趾和焊根处变化趋势相似.

![](images/8281f7e9d378e1f0ceb67638d9bb680d13664e75b8d320cd5399194cf58bdd91.jpg)  
图 7 应变能密度与焊脚和熔深的关系 Fig. 7 Relationship between strain energy density and weld size and penetration. (a) weld toe; (b) weld root

为了进一步分析焊脚和熔深的影响，根据Xing 等人[17] 的建议，取熔深为 0 和 0.2 时的数据，如图 8 所示.从图 8a 中可以看出，熔深为 0.2 时，相对焊脚尺寸小于 0.64 时，焊根处的应变能密度大于焊趾处的应变能密度，此时认为焊根为疲劳失效位置；当相对焊脚尺寸大于 0.64 时，焊根处的应变能密度小于焊趾处的应变能密度，此时认为焊趾是疲劳失效位置.从图 8b 中可以看出，当熔深为0 时，焊根和焊趾处应变能密度曲线交点为 s/t =1.12.综合图 8 可以得到基于平均应变能密度的疲劳失效位置转换区间，如图 9 所示.在区间左侧，均为焊根失效，在区间右侧均为焊趾失效，在区间内部既存在焊根失效也存在焊趾失效.

为验证上述转换区间的准确性，截取了文献 [17] 的部分疲劳数据，其采用十字接头，板厚固定为 5 mm 和 10 mm，焊脚尺寸 3 \~ 12 mm 不等；疲劳试验分两组，分别施加 207 MPa 和 414 MPa的载荷[17].将其疲劳数据和文中得到的转换区间结合，如图 10 所示.从图中可以看出，不论是施加 207 MPa 的载荷(图 10a) 还是 414 MPa 的载荷(图 10b)，在转换区间的左侧都是焊根失效的疲劳点，在转换区间右侧都是焊趾失效的疲劳点，而在区间内部既有焊根失效也有焊趾失效.这验证了文中基于应变能密度得到的疲劳失效位置转换区间的正确性.

![](images/cd8bfdd5f977990f860d563bd8079bb2e549372183210065e8cc80f3737a87f5.jpg)  
图 8 固定熔深下的应变能密度

![](images/5d2aad9088504fe23d105a2808f29134212c4ee29190ab8f818ef42eb0463df4.jpg)  
Fig. 8 Strain energy density at fixed penetration. (a) the penetration is 0.2; (b) the penetration is 0   
图 9 转换区间Fig. 9 Transition interval

![](images/f7855c27c093bed0d3d5878ea8981b7b673df7c3cd93367dd4b29bcabc184638.jpg)  
图 10 结合疲劳数据的失效位置转换区间  
Fig. 10 Failure position transition interval combined with fatigue data. (a) tensile load P = 207 MPa; (b) tensile load P = 414 MPa

# 4    结论

(1) 局部应变能密度法既可用于焊根失效疲劳评估，也可以用于焊趾失效疲劳评估.不同位置的疲劳裂纹会影响结构的刚度，进而影响结构的疲劳寿命.(2) 焊接接头的焊脚尺寸和熔深影响接头的疲劳性能.焊根和焊趾处的应变能密度值均随着焊脚和熔深的增大而减小.(3) 得到了基于应变能密度的疲劳失效位置转换区间，并通过疲劳数据验证了其合理和可行性.接头熔深在 0 \~ 0.2 之间时，疲劳失效位置转换区间对应的相对焊脚尺寸是 0.64 \~ 1.12.

# 参考文献

[1] 霍立兴. 焊接结构的断裂行为及评定 [M]. 北京: 中国建筑工业 出版社, 2000. Huo Lixing. Fracture behavior and evaluation of welded structure[M]. Beijing: China Architecture & Building Press, 2000.   
[2] 王文先. 焊接结构 [M]. 北京: 化学工业出版社, 2018. Wang Wenxian. Welding structure[M]. Beijing: Chemical Industry Press, 2018.   
[3] Radaj D. Review of  fatigue strength assessment of  non-welded and welded structures based on local  parameters[J]. International Journal of Fatigue, 1996, 18(3): 153 − 170.   
[4] Frick W. Review of fatigue analysis of welded joints: state of development[J]. Marine Structure, 2003, 16: 185 − 200.   
[5] Hobbacher A. Recommendations for fatigue design of welded joints and components[M]. Springer International Publishing, 2016.   
[6] Veritas D  N. Fatigue  design of  offshore  steel  structures[S]. Norway: DNV Recommended Practice DNVGL-RP-C203, 2016.   
[7] Lazzarin P. A finite-volume-energy based approach to predict the static andfatigue behavior of  components with sharp V-shaped notches[J]. International Journal of  Fatigue,  2001, 112(3): 275 − 298.   
[8] Williams M  L. Stress singularities resulting  from various boundary conditions in angular corners of plates in tension[J]. Journal of Applied Mechanics, 1952, 19: 526 528.   
[9] Gross B, Mendelson A. Plane elastostatic  analysis of  V-notched plates[J]. International Journal of Fracture Mecanics, 1972, 8: 267 − 276.   
[10] Zappalorto  M,  Lazzarin P,  Yates J  R. Elastic stress distributions for hyperbolic and parabolic notches in round shafts under torsion and uniform antiplane shear loadings[J]. International Journal of Solids and Structures, 2008, 45: 4879 − 4901.   
[11] Lazzarin P, Tovo R.  A unified approach to the evaluation of   linear elastic stress fields in the neighborhood of cracks and notches[J]. International Journal of Fracture, 1996, 78: 3 − 19.   
[12] Livieri P, Lazzarin P. Fatigue strength of steel and aluminium welded jointsbased on generalised stress intensity factors and local strain energy values[J]. International Journal of Fracture Mechanics, 2005, 133: 247 − 76.   
[13] Lazzarin P, Berto  F, Gómez F  J. Some advantages derived from the use of  the strain energy density over a control volume in fatigue strength assessments of welded joints[J]. International Journal of Fatigue, 2008, 30: 1345 − 1357.   
[14] Berto F, Lazzarin P. A review of the volume-based strain energy density  approach applied to V-notches and welded structures[J]. Theoretical and Applied Fracture Mechanics, 2009, 52: 183 194.   
[15] Wang D, Zhang H, Gong B. Residual stress effects on fatigue behaviour of welded T-joint: A finite fracture mechanics approach[J]. Materials & Design, 2016, 91: 211 − 217.   
[16] Kihl P and Sarkani S. Thickness effects on the fatigue strength of welded steel cruciform[J]. International Journal of Fatigue, 1997, 19: 311 − 316.   
[17] Xing S, Dong  P, Threstha A. Analysis of  fatigue failure mode transition in load-carrying fillet-welded connections[J]. Marine Structures, 2016, 46: 102 − 126.

第一作者简介：刘夕，1986 年出生，硕士研究生.主要从事焊接结构疲劳方面研究工作. Email： beyondlx@tju. edu.cn.

（编辑： 周珍珍）

# [ 上接第 59 页 ]

[22] Pisarev V, Odintsev I, Eleonsky S, et al. Residual stress determination by optical interferometric measurements of hole diameter increments[J]. Optics and Lasers in Engineering, 2018, 110: 437 456.   
[23] Shokrieh M M, Jalili S  M, Kamangar M A. An eigen-strain approach on the estimation of non-uniform residual stress distribution using incremental hole-drilling and slitting techniques[J]. International Journal of Mechanical Sciences, 2018, 148: 383 − 392.   
[24] 张清东, 陈先霖, 王长松, 等. 冷轧宽带钢横向内应力分布的试 测与计算[J]. 北京科技大学学报, 1994(s2): 81 − 85. Zhang Qingdong, Chen Xianlin, Wang  Changsong, et al. Measurement and calculation of transversal internal stress distribution in the Off-line cold rolled strip[J]. Journal of  University of Science and Technology Beijing, 1994(s2): 81 − 85.   
[25] 李博, 张清东, 张晓峰. 带钢平整轧制残余应力场的二维数值模 拟 [J]. 轧钢, 2014, 31(1): 14 − 18. Li Bo, Zhang Qingdong, Zhang Xiaofeng, et al. Two-dimensional numerical simulation of  residual stress of strip in temper rolling process[J]. Steel Rolling, 2014, 31(1): 14 − 18.   
[26] 翟传明, 邸小坛, 白伟亮, 等. 盲孔法检测既有金属结构应力的 研究 [J]. 建筑科学, 2011, 27(s1): 116 − 120. Zhai Chuanming, Di Xiaotan, Bai Weiliang, et  al. Research on stress inspected by blind hole method in the existing metal structures[J]. Bulding Science, 2011, 27(s1): 116 − 120.

（编辑： 周珍珍）

Car Co., Ltd, Changchun, 130012, China；6. Weihai Donghai Shipyard CO. LTD, Weihai, 264209, China). pp 69-73

Abstract: The Ag-Cu-Ti filler metal was wetted on C/C composites by the TIG arc directly. The macroscopic features of the wetting filler metal under different holding times were observed. Besides, the microstructure and element distribution of the joints was analyzed by SEM and EDS, respectively. The experimental results showed that the filler metal can soften under certain holding time. When the holding time was 60 min, the wetting effect of  filler metal on C/C composites was the best. The distribution of the TiC reaction layer was the most uniform and dense with a thickness of 1.3 μm. Besides, the maximum thickness of the diffusion layer was 5.5 μm. When the content of copper and silver in the filler metal approaching to the reaction interface remained the same, there were brittle compounds such as AgTi/CuTi3/Cu4Ti3. Meanwhile, Ti elements gathered near the interface and Ti2Cu was produced in the aggregation area.

Key words: TIG arc； Ag-Cu-Ti filler metal；C/C composites； wetting

Fatigue performance analysis of weld toe and root of thick T-joint based on local strain energy density LIU Xi1， CHEN Guangran ， MENG Qingyu1， GONG Baoming2， WANG Can1 (1. Jiangsu XCMG Construction Machinery Research Institute Ltd., Xuzhou, 221004, China； 2. Tianjin University, Tianjin, 300072, China). pp 74-80

Abstract: In this paper, for the purpose of evaluating the fatigue life of the loader’s front frame, three typical welded joints are extracted from the key parts of the welded structure of the front frame, and the fatigue test and analysis are carried out. Based on the local strain energy density method, the fatigue life curve is obtained through the experimental data, the accuracy and applicability of the theory are then verified by the fatigue data of the cross joint in the literature. The results show that the local strain energy density method can be used for the fatigue assessment of weld root and weld toe failure. Fatigue cracks at different positions will affect the rigidity of the structure, and thus the fatigue life of the structure. The influence of the size of the weld foot and the penetration depth of the joint on the fatigue life of the welded joint is studied, then derived the fatigue failure position transition interval from the local strain energy density method. The accuracy of  the transition interval between weld toe failure and the root failure was verified by the fatigue data in the literature, which provides a basis for the fatigue design of partial penetration Tjoints.

Key words: fatigue assessment； finite element； strain energy density

Effects of MIG welding superposition on microstructure and property of 6A01-T5 FSW joint ZHANG Tiehao1， YANG Zhibin1,2， ZHANG Zhiyi1， ZHANG Haijun ， Shi Chunyuan2 (1. CRRC Qiangdao Sifang Co., Ltd., Qiangdao 266111, China； 2. School of Material Science and Engineering, Dalian Jiaotong University, Dalian 116028, China). pp 81-88,96

Abstract: Effects of  MIG welding superposition on microstructure and property of 6A01-T5 FSW joint was researched in this work. The results indicated that the MIG/FSW joints formed well without porosity defects near the superposition. The microstructure of the FSW weld nugget and heat affected zone became coarse and which near the superposition was changed obviously. The hardness value of the superposition was reduced significantly, especially for FSW thermo-mechanically affected zone and heat affected zone. The tensile strengths of the FSW joint, MIG superposition on the FSW weld center, MIG superposition on the FSW advancing side thermo-mechanically affected zone,  and MIG superposition on the FSW retreating side thermo-mechanically affected zone were 219.8 MPa, 188.0 MPa, 195.4 MPa and 191.4 MPa，respectively. MIG superposition reduced the FSW joint tensile strength, and the fracture appearance of all joints belonged to ductile fracture. The median fatigue strengths of above the FSW joint and three joints with MIG weld reinforcement were 76.7 MPa, 65.0 MPa, 67.5 MPa and 65.0 MPa respectively. The MIG superposition was also reduced the fatigue properties of the FSW joints.

Key words: Aluminum alloy； Frication stir welding； Superposition welding；Mechanical property ；MIG welding
