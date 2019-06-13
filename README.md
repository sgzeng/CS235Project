# CS235Project
UCR CS235 Data Mining Group Project

## English Description

This is the final project of UCR CS235. Team members: Haochen Zeng, Mingchen Li, Yujia Zhai.

- Raw data: 16000 white, 8000 black (training), 4000 white, 2000 black (testing).

- Fused feature extraction scheme.

- LogReg, NN, testing accuracy up to 94%.

# Ref:

## Related Methodology
https://xz.aliyun.com/t/3704

## Priori Knowledges
A list of Sensitive Windows API: p453 <<Practical Malware Analysis>>

https://esebanana.github.io/2018/04/28/re_15_evil_3/

https://github.com/rshipp/awesome-malware-analysis/blob/master/%E6%81%B6%E6%84%8F%E8%BD%AF%E4%BB%B6%E5%88%86%E6%9E%90%E5%A4%A7%E5%90%88%E9%9B%86.md

# Chinese Description
## 大数据比赛-恶意代码识别分赛第一题说明文档

1. 第一题题目设计说明
面对大量的 PC 恶意病毒的动态行为进行识别，进行分类判别，即判断一个 exe 文件 经过沙箱运行后输出的 x􏰀􏰁 文件来判断该文件是否为恶意程序。
分析程序的种类，需要完成:根据训练集样本，对测试集样本进行是否是恶意样本的 判断。
Safe_ty􏰂e: 安全结果，恶意程序为 1，正常程序为 0

2. 题目数据的下发形式及文件大小(初赛数据已完成，决赛数据还在进行筛选)
(1) 数据集说明(第一题):
约 45000 条样本数据，压缩包 􏰀d5 值为 210d2b05a3a5afc3f1458d408f802820; 训练数据集约为 30000 条(其中黑样本 10000 条，白样本 20000 条，并包含样 本 gr􏰃u􏰄d truth 标签)测试数据集为 15000 条，文件类型为.x􏰀􏰁。

3. 总体题目结果提交形式及文件大小
(1) (2) (3)
第一题比赛期间提交识别结果 CSV 文件(CSV 文件见示例)，最后提交算法程 序
最终比赛成绩结果按 Sc􏰃reF= 0.5*Sc􏰃re1 题目一 + 0.5*Sc􏰃re2 题目二进行排名， 并提交第一题及第二题的算法设计说明 PPT
经评委对决赛成绩进行审核后，确认无作弊及其他异常问题后，最终颁布决赛 排名成绩

4. 题目评判规则 (1) 评分规则
N: 测试集样本数量
𝑆𝑇 :第 k 个样本的 Safe_type 预测值
𝑇𝑟𝑢𝑒𝑆𝑇 :第 k 个样本的 Safe_type 标准值 -
[ ]:判定成立为 1，判定不成立为 0
Safe_type 判断正确，可得 1 分;
Safe_type 判断不正确时。若属于误报，即白样本判断为恶意样本，扣 1 分。
若属于漏报，即恶意样本判断为白样本，不得分也不扣分。 备注:更看重对误报的惩罚，参赛者提交的是非 0 即 1 的硬分类结果。
-(2) 提交说明:第一题比赛每日可提交一次，取截至日期前得分最高成绩; 决赛成 绩按截至日期前最优成绩排名确定(进入前三名的队伍最终的名次确定会参考 一定比例的算法设计思路).
