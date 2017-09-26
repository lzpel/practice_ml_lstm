import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by user on 2017/07/13.
 */
public class Main {
	public static void main(String[] args) {
		test_CHAR_RNN();
	}
	static void test_CHAR_RNN(){
		NNM_LSTM nnm=new NNM_LSTM(16,500,16, NNL_OUT.ACTIVE_SIGM,30);
		try{
			for(int t=0;t<1000;t++) {
				BufferedReader br = new BufferedReader(new FileReader("src\\sample.txt"));
				for (int a = 0; (a = br.read()) != -1; ) {
					for (int i = 0; i < nnm.getd().length; i++) {
						nnm.geti()[i] = nnm.getd()[i];
					}
					for (int i = 0; i < nnm.getd().length - nnm.so; i++) {
						nnm.getd()[i] = nnm.getd()[i + nnm.so];
					}
					for (int i = 0; i < nnm.so; i++) {
						nnm.getd()[nnm.so * (nnm.l - 1) + i] = ((a & (1 << i)) != 0) ? 1f : 0f;
					}
					nnm.learn(0.1f);
					int c = 0;
					for (int j = 0; j < nnm.so; j++) {
						if (nnm.geto()[nnm.so * (nnm.l - 1) + j] > 0.5) {
							c += 1 << j;
						}
					}
					System.out.print((char) c);
				}
				System.out.println();
				br.close();
			}
		}catch(FileNotFoundException e){
			System.out.println(e);
		}catch(IOException e){
			System.out.println(e);
		}
		/*
		for(int m=0;true;m++){
			for(int n=0;n<nnm.l;n++){
				for(int x=0;x<nnm.si;x++){
					nnm.geti()[n*nnm.si+x]=nnm.getd()[n*nnm.si+x];
					nnm.getd()[n*nnm.si+x]=
				}
			}
			nnm.learn(0.01f);
		}
		for(int m=0;m<100;m++){
			nnm.geti()[0]=nnm.geto()[0];
			nnm.learn();
			System.out.println(nnm.geto()[0]);
		}
		*/
	}
	void test_NNVNNN(){
		int type=2;
		float loss=0;
		NNV v0=new NNV(0);
		NNV v1=new NNV(2);
		NNV v2=new NNV(2);
		NNV v3=new NNV(2);
		NNN n=new NNN(0,2);
		v3.v[0]=0;
		v3.v[1]=1;
		for(int i=0;i<100;i++) {
			//順伝播
			NNV.vcross(true, v1, v0, n);
			switch (type) {
				case 0:
					NNV.videnactive(true, v2, v1);
					loss = NNV.videnloss(v3, v2, v1);
					break;
				case 1:
					NNV.vsigmactive(true, v2, v1);
					loss = NNV.vsigmloss(v3, v2, v1);
					break;
				case 2:
					NNV.vsmaxactive(true, v2, v1);
					loss = NNV.vsmaxloss(v3, v2, v1);
					break;
			}
			//逆伝播
			NNV.vcross(false, v1, v0, n);
			//修正
			NNV.nlearn(v1, v0, n,1f);
		}
	}
}
