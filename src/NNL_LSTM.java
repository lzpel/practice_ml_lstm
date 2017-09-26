/**
 * Created by user on 2017/08/22.
 * LSTMの層を表す
 */
public class NNL_LSTM extends NNL{
	NNN Rz,Ri,Ro,Rf,Wz,Wi,Wo,Wf;
	NNV vz[],vi[],vo[],vf[],vc[],vx[];
	NNL_LSTM(NNL m){
		super(m.si,m.so);
		initvector();
	}
	NNL_LSTM(int size_i,int size_o){
		super(size_i,size_o);
		initvector();
		Rz = new NNN(so, so);
		Ri = new NNN(so, so);
		Ro = new NNN(so, so);
		Rf = new NNN(so, so);
		Wz = new NNN(si, so);
		Wi = new NNN(si, so);
		Wo = new NNN(si, so);
		Wf = new NNN(si, so);
	}
	public void initvector(){
		vx=NNV.newmulti(5,si);
		vz=NNV.newmulti(4,so);
		vi=NNV.newmulti(4,so);
		vo=NNV.newmulti(4,so);
		vf=NNV.newmulti(4,so);
		vc=NNV.newmulti(10,so);
	}
	public NNV geti(){
		return vx[0];
	}
	public NNV geto(){
		return vc[5];
	}
	public void propagate(boolean f){
		if(f){
			//x
			NNV.vcpy(f,vx[1],vx[2],vx[3],vx[4],null,vx[0]);
			//z
			NNV.vcross(f,vz[0],vx[1],Wz);
			NNV.vcross(f,vz[1],vc[6],Rz);
			NNV.vsum(f,vz[2],vz[0],vz[1]);
			NNV.vtanhactive(f,vz[3],vz[2]);
			//i
			NNV.vcross(f,vi[0],vx[2],Wi);
			NNV.vcross(f,vi[1],vc[7],Ri);
			NNV.vsum(f,vi[2],vi[0],vi[1]);
			NNV.vsigmactive(f,vi[3],vi[2]);
			//o
			NNV.vcross(f,vo[0],vx[3],Wo);
			NNV.vcross(f,vo[1],vc[8],Ro);
			NNV.vsum(f,vo[2],vo[0],vo[1]);
			NNV.vsigmactive(f,vo[3],vo[2]);
			//f
			NNV.vcross(f,vf[0],vx[4],Wf);
			NNV.vcross(f,vf[1],vc[9],Rf);
			NNV.vsum(f,vf[2],vf[0],vf[1]);
			NNV.vsigmactive(f,vf[3],vf[2]);
			//c
			NNV.vmul(f,vc[1],vc[0],vf[3]);
			NNV.vmul(f,vc[2],vz[3],vi[3]);
			NNV.vsum(f,vc[0],vc[1],vc[2]);
			NNV.vtanhactive(f,vc[3],vc[0]);
			NNV.vmul(f,vc[4],vc[3],vo[3]);
			NNV.vcpy(f,vc[5],vc[6],vc[7],vc[8],vc[9],vc[4]);
		}else{
			NNV.vcpy(f,vc[5],vc[6],vc[7],vc[8],vc[9],vc[4]);
			NNV.vmul(f,vc[4],vc[3],vo[3]);
			NNV.vtanhactive(f,vc[3],vc[0]);
			NNV.vsum(f,vc[0],vc[1],vc[2]);
			NNV.vmul(f,vc[2],vz[3],vi[3]);
			NNV.vmul(f,vc[1],vc[0],vf[3]);
			//c
			NNV.vsigmactive(f,vf[3],vf[2]);
			NNV.vsum(f,vf[2],vf[0],vf[1]);
			NNV.vcross(f,vf[1],vc[9],Rf);
			NNV.vcross(f,vf[0],vx[4],Wf);
			//f
			NNV.vsigmactive(f,vo[3],vo[2]);
			NNV.vsum(f,vo[2],vo[0],vo[1]);
			NNV.vcross(f,vo[1],vc[8],Ro);
			NNV.vcross(f,vo[0],vx[3],Wo);
			//o
			NNV.vsigmactive(f,vi[3],vi[2]);
			NNV.vsum(f,vi[2],vi[0],vi[1]);
			NNV.vcross(f,vi[1],vc[7],Ri);
			NNV.vcross(f,vi[0],vx[2],Wi);
			//i
			NNV.vtanhactive(f,vz[3],vz[2]);
			NNV.vsum(f,vz[2],vz[0],vz[1]);
			NNV.vcross(f,vz[1],vc[6],Rz);
			NNV.vcross(f,vz[0],vx[1],Wz);
			//z
			NNV.vcpy(f,vx[1],vx[2],vx[3],vx[4],null,vx[0]);
			//x
		}
	}
	public void learn(float r){
		NNV.nlearn(vz[0],vx[1],Wz,r);
		NNV.nlearn(vi[0],vx[2],Wz,r);
		NNV.nlearn(vo[0],vx[3],Wz,r);
		NNV.nlearn(vf[0],vx[4],Wz,r);
		NNV.nlearn(vz[1],vc[6],Rz,r);
		NNV.nlearn(vi[1],vc[7],Ri,r);
		NNV.nlearn(vo[1],vc[8],Ri,r);
		NNV.nlearn(vf[1],vc[9],Ri,r);
	}
	public void zerodelta(){
		for(int n=0;n<vz.length;n++)vz[n].zerodelta();
		for(int n=0;n<vi.length;n++)vi[n].zerodelta();
		for(int n=0;n<vo.length;n++)vo[n].zerodelta();
		for(int n=0;n<vf.length;n++)vf[n].zerodelta();
		for(int n=0;n<vx.length;n++)vx[n].zerodelta();
		for(int n=0;n<vc.length;n++)vc[n].zerodelta();
	}
	public void zerovalue(){
		for(int n=0;n<vz.length;n++)vz[n].zerovalue();
		for(int n=0;n<vi.length;n++)vi[n].zerovalue();
		for(int n=0;n<vo.length;n++)vo[n].zerovalue();
		for(int n=0;n<vf.length;n++)vf[n].zerovalue();
		for(int n=0;n<vx.length;n++)vx[n].zerovalue();
		for(int n=0;n<vc.length;n++)vc[n].zerovalue();
	}
	public void lcpy(boolean f,NNL i0){
		NNL_LSTM i=(NNL_LSTM)i0;
		for(int n=0;n<vz.length;n++)NNV.vcpy(f,vz[n],i.vz[n]);
		for(int n=0;n<vi.length;n++)NNV.vcpy(f,vi[n],i.vi[n]);
		for(int n=0;n<vo.length;n++)NNV.vcpy(f,vo[n],i.vo[n]);
		for(int n=0;n<vf.length;n++)NNV.vcpy(f,vf[n],i.vf[n]);
		for(int n=0;n<vx.length;n++)NNV.vcpy(f,vx[n],i.vx[n]);
		for(int n=0;n<vc.length;n++)NNV.vcpy(f,vc[n],i.vc[n]);
	}
}