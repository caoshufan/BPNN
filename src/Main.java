import java.util.*;

public class Main
{
	public static double transform(double value, double min, double max) {
		double a = 0.9;
		double b = (1 - a) / 2;
		return (value - min) / (max - min) * a + b;
	}
	
	public static double untransform(double value, double min, double max) {
		double a = 0.9;
		double b = (1 - a) / 2;
		return (value - b) / a * (max - min) + min;
	}
	
	public static void main(String[] args)
	{
		int inputCount = 4;
		int hiddenCount = 9;
		int outputCount = 3;
		BP bp = new BP(inputCount, hiddenCount, outputCount, 0.25, 0.9);
		double[] data = {2378.9, 2476.8, 2706.5, 2413.3, 2585.6, 2637.2, 2596.3, 2784.5, 2618.4, 2896.7, 3035.3, 3266.3, 3304.2};
		double min = data[0];
		double max = data[0];
		for(int i = 1; i < data.length; i++) {
			if(min > data[i]) min = data[i];
			if(max < data[i]) max = data[i];
		}
		
		double[][] inputData = new double[data.length - inputCount][inputCount];
		double[][] outputData = new double[data.length - inputCount][outputCount];
		
		for(int i = 0; i < data.length - inputCount; i++) {
			for(int j = 0; j < inputCount; j++)
				inputData[i][j] = transform(data[i+j], min, max);
			for(int j = inputCount; j < inputCount + outputCount; j++)
				outputData[i][j - inputCount] = transform(data[i+j-outputCount+1], min, max);
		}
		
		Random random = new Random();
		int times = 0;

		do {
			int idx = random.nextInt(data.length - inputCount - 1);
			bp.train(inputData[idx], outputData[idx]);
			System.out.format("第%d次训练: %f\n", ++times, bp.errorSum);
		} while(bp.errorSum > 0.001);
		
		System.out.println("训练完毕!!");
		
		for(int i = 0; i < data.length - inputCount; i++) {
			double[] output = bp.test(inputData[i]);
			double f = untransform(output[outputCount - 1], min, max);
			double a = data[i + inputCount];
			System.out.format("第%d个测试样例：\n", i+1);
			System.out.println("预测值: " + f);
			System.out.println("实际值: " + a);
		}
	}
}
