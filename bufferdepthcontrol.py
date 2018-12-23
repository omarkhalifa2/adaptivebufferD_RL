import numpy as np
import buffermodule # buffer module reponsible for all interactions with the buffer

N=300
I=50
delay_ancien=0
steps=100
lr=0.001
batch_size = 10 
decay=0.99


neurnet={}
neurnet["w1"]=np.rand(300,50)/np.sqrt(50)
neurnet["w2"]=np.rand(300)/np.sqrt(300)

wei_buff={i:np.zeros(j)  for i,j in neurnet.items() }
rmsprop_cache = { k : np.zeros_like(v) for k,v in neurnet.items() } 

def sigmoid(x):
	return 1/(1+np.exp(-x))

def forwardp(i):
	h=np.dot(neurnet["w1"],i)
	h[h<0]=0
	o=np.dot(neurnet["w1"],h)
	o=sigmoid(o)
	return h,o

def prepo(dl):
	I[1:]=I[:-1]
	I[0]=dl
	return I

def backprop(grad,ninput,h):
	dw2=np.dot(h.T,grad).ravel()
	buff=np.outer(grad,neurnet("w2"))
	buff[h<=0]=0
	dw1=np.dot(npbuff.T,ninput)
	return {"w1":dw1,"w2":dw2}

def policy(vector):
	sum=0
	ovector=np.zeros_like(vector)
	for i in range(0,vector.size,-1)
		if vector[i]!=0:
			sum=vector[i]
			break
		else:
			sum*=0.9
		ovector[i]=sum

	return ovector



grad,inpl,hl,feedbackl=[],[],[],[]
delay=buffermodule.getdelai() # gives us the the delay of this packet
episode_num=0
action=0

while True:


	i=delay-delay_ancien
	netinput=prepo(i)
	inpl.append(netinput)

	netoutput,h=forwardp(netinput)
	hl.append(h)

	action = 1 if np.random.uniform() < netoutput else -1 
	y=1 if action == 1 else 0
	grad.append(y-netoutput)

	feedback,completed=buffermodule.change(action) #changes the size of the buffer 
	feedbackl.append(feedback)

	if completed == True:#end of an episode which reprensent a number of packets between each quality evaluation

		hiddens=np.vstack(hl)
		inputs=np.vstack(inpl)
		feedbacks=np.vstack(feedbackl)
		grads=np.vstack(grad)
		grad,inpl,hl,feedbackl=[],[],[],[]

		feedbacks=policy(feedbacks)

		feedbacks-= np.mean(feedbacks)
		feedbacks/= np.std(feedbacks)

		grads=grads*feedbacks
		dictofgrad=backprop(grads,inputs,hiddens)

		for k in neurnet: wei_buff[k] += dictofgrad[k]

		if episode_num % batch_size == 0

			for k,v in neurnet.items():
				g=wei_buff[k]
				rmsprop_cache[k]=decay*rmsprop_cache[k]+(1 - decay) * g**2
				neurnet[k]+= lr * g/ (np.sqrt(rmsprop_cache[k])+ 1e-5)
		