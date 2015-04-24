/**
 * BP神经网络
 * @author dog
 * 实现多层前馈网络算法
 */
public class BP {
	/**
	 * input[1..inputCount] : 输入值
	 * input[0] = -1 : 隐含层阀值
	 */
	private double[] input;
	/**
	 * hidden[1..hiddenCount] : 隐含层输出值
	 * hidden[0] = -1 : 输出层阀值
	 */
	private double[] hidden;
	/**
	 * output[1..outputCount] : 输出值
	 * output[0] = 0 : 没作用
	 */
	private double[] output;
	/**
	 * actOutput[1..actOutputCount] : 实际输出值
	 * actOutput[0] = 0 : 没作用
	 */
	private double[] actOutput;
	/**
	 * 输入层与隐含层间的权值矩阵
	 * iptHidWeight[1..inputCount][j](j=1..hiddenCount) : 隐含层第j个节点的接入权值
	 * input[0] * iptHidWeight[0][j](j=1..hiddenCount) : 隐含层第j个节点修改后的阀值
	 */
	private double[][] iptHidWeights;
	/**
	 * 隐含层与输出层间的权值矩阵
	 * hidOptWeight[1..hiddenCount][j](j=1..outputCount) : 输出层第j个节点的接入权值
	 * hidden[0] * hidOptWeight[0][j](j=1..outputCount) : 输出层第j个节点修改后的阀值
	 */
	private double[][] hidOptWeights;
	/**
	 * 输入层与隐含层间的权值矩阵修改量
	 */
	private double[][] iptHidWeightsDelta;
	/**
	 * 隐含层与输出层间的权值矩阵修改量
	 */
	private double[][] hidOptWeightsDelta;
	/**
	 * optError[i](i=1..outputCount) : 第i个输出元的误差信号
	 */
	private double[] optError;
	/**
	 * hidError[i](i=1..hiddenCount) : 第i个隐含元的误差信
	 */
	private double[] hidError;
	/**
	 * 学习效率
	 */
	public double tau;
	/**
	 * 动量
	 */
	private double momentum;
	/**
	 * 总误差
	 */
	public double errorSum;

//	/**
//	 * error[i](i=0..caseCount-1) : 第i个训练样本的总误差值
//	 * 在计算总体误差时用到
//	 */
//	private double[] error;

	/**
	 * 初始化神经网络
	 * @param inputCount 输入元数目
	 * @param hiddenCount 隐含层节点数
	 * @param outputCount 输出元数目
	 * @param tau 学习效率
	 * @param momentum 动量
	 */
	public BP(int inputCount, int hiddenCount, int outputCount, double tau, double momentum) {
		this.input = new double[inputCount + 1];
		this.hidden = new double[hiddenCount + 1];
		this.output = new double[outputCount + 1];
		this.actOutput = new double[outputCount + 1];
		this.iptHidWeights = new double[inputCount + 1][hiddenCount + 1];
		this.hidOptWeights = new double[hiddenCount + 1][outputCount + 1];
		this.iptHidWeightsDelta = new double[inputCount + 1][hiddenCount + 1];
		this.hidOptWeightsDelta = new double[hiddenCount + 1][outputCount + 1];
		this.optError = new double[outputCount + 1];
		this.hidError = new double[hiddenCount + 1];
		this.tau = tau;
		this.momentum = momentum;

		input[0] = -1;
		hidden[0] = -1;
		output[0] = 0;
		actOutput[0] = 0;

		errorSum = 0;

		initWeightMatrix();
	}

	/**
	 * 初始化权值矩阵
	 */
	private void initWeightMatrix() {
		int inputCount = input.length - 1;
		int hiddenCount = hidden.length - 1;
		int outputCount = output.length - 1;

		//隐含层初始化(净输入在零点附近，变化处于敏感区域，可使学习速度较快)
		for(int i = 0; i <= inputCount; i++) {
			for(int j = 1; j <= hiddenCount; j++) {
				iptHidWeights[i][j] = 0;
			}
		}

		//输出层初始化(如果初始权值太小会使隐含层调整量变小)
		for(int i = 0; i <= hiddenCount; i++) {
			for(int j = 1; j <= outputCount; j++) {
				hidOptWeights[i][j] = i%2==0 ? -1 : 1;
			}
		}
	}

	/**
	 * 传入一组样例，进行一次训练
	 */
	public void train(double[] dataInput, double[] dataOutput) {
		System.arraycopy(dataInput, 0, input, 1, dataInput.length);
		System.arraycopy(dataOutput, 0, actOutput, 1, dataOutput.length);

		forward(input, hidden, iptHidWeights);
		forward(hidden, output, hidOptWeights);
		computeError();
		maintainWeights();
	}

	/**
	 * 进行神经网络的测试
	 */
	public double[] test(double[] dataInput) {
		System.arraycopy(dataInput, 0, input, 1, dataInput.length);

		forward(input, hidden, iptHidWeights);
		forward(hidden, output, hidOptWeights);
		return output;
	}

	/**
	 * 计算该层的输出值
	 */
	private void forward(double[] input, double[] output, double[][] weight) {
		int inputCount = input.length - 1;
		int outputCount = output.length - 1;

		for(int j = 1; j <= outputCount; j++) {
			output[j] = 0;
			for(int i = 0; i <= inputCount; i++) {
				output[j] += input[i] * weight[i][j];
			}
			output[j] = sigmoid(output[j]);
		}
	}

	/**
	 * 计算误差信号
	 */
	private void computeError() {
		int hiddenCount = hidden.length - 1;
		int outputCount = output.length - 1;
		errorSum = 0;

		//计算输出层错误信号
		for(int i = 1; i <= outputCount; i++) {
			optError[i] = (actOutput[i] - output[i]) * output[i] * (1 - output[i]);
			errorSum += Math.abs(optError[i]);
		}

		//计算隐含层错误信号
		for(int i = 1; i <= hiddenCount; i++) {
			hidError[i] = 0;
			for(int j = 1; j <= outputCount; j++) {
				hidError[i] += optError[j] * hidOptWeights[i][j] * hidden[i] * (1 - hidden[i]);
			}
		}
	}

	/**
	 * 修正权值
	 */
	private void maintainWeights() {
		int inputCount = input.length - 1;
		int hiddenCount = hidden.length - 1;
		int outputCount = output.length - 1;

		//计算隐含层与输出层间权值矩阵修改量(增加动量项)
		for(int i = 0; i <= hiddenCount; i++) {
			for(int j = 1; j <= outputCount; j++) {
				hidOptWeightsDelta[i][j] = (tau * optError[j] * hidden[i])
					+ (momentum * hidOptWeightsDelta[i][j]);

				hidOptWeights[i][j] += hidOptWeightsDelta[i][j];
			}
		}

		//计算输入层与隐含层间权值矩阵修改量(增加动量项)
		for(int i = 0; i <= inputCount; i++) {
			for(int j = 1; j <= hiddenCount; j++) {
				iptHidWeightsDelta[i][j] = (tau * hidError[j] * input[i])
					+ (momentum * iptHidWeightsDelta[i][j]);

				iptHidWeights[i][j] += iptHidWeightsDelta[i][j];
			}
		}
	}

	/**
	 * sigmoid函数
	 * @param x
	 * @return
	 */
	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
}
