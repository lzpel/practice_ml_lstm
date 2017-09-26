/**
 * Created by user on 2017/08/21.
 * 重みを管理
 */
public class NNN {
	float v[];
	int si,so;
	NNN(int size_i,int size_o){
		si=size_i;
		so=size_o;
		v=new float[(si+1)*so];
		setrandom();
	}
	void setrandom(){
		for(int i=0;i<v.length;i++){
			v[i]=(float)Math.random()-0.5f;
		}
	}
}
