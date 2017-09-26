/**
 * Created by user on 2017/08/25.
 */
public class NNM {
	int si,sm,so;
	NNM(int size_i,int size_m,int size_o){
		si=size_i;
		sm=size_m;
		so=size_o;
	}
	public float learn(){
		return learn(-1f);
	}
	public float learn(float rate){
		return 0;//loss
	}
	public float[] geti(){
		return null;
	}
	public float[] geto(){
		return null;
	}
	public float[] getd(){
		return null;
	}
}