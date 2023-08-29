import torch

class preprocess:
    
    def __init__(self,independent_pairs, asymmetric_pairs = True):
        """
        independent pairs : True = the sequence (Zi,Zj) is comprised of independent copies (length n/2)
        asymmetric pairs : True = (Zi,Zj)_i<j, False = (Zi,Zj)i!=j
        """
        self.independent_pairs = independent_pairs
        self.asymmetric_pairs = asymmetric_pairs
        
    def __call__(self,X,Y):
        if self.independent_pairs:
            n = int(len(Y)/2)
            Inputs,Outputs = (torch.column_stack((torch.row_stack((X[n:],X[:n])),
                                                  torch.row_stack((X[:n],X[n:])),
                                                  torch.row_stack((Y[:n],Y[n:])))),
                                            torch.row_stack((Y[n:],Y[:n]))) 
        else:
            n,P = Y.size()
            D = len(X.T)
            m = n*(n-1)
            if self.asymmetric_pairs:
                m = int(m/2)
            
            Xcombi,Xcombj = torch.zeros((m,D)), torch.zeros((m,D))
            Ycombi,Ycombj = torch.zeros((m,P)), torch.zeros((m,P))
            for d in range(D):
                Xcomb = torch.combinations(X[:,d])
                if not self.asymmetric_pairs:
                    Xcomb = torch.column_stack((torch.row_stack((Xcomb[:,0],Xcomb[:,1])),
                                                torch.row_stack((Xcomb[:,1],Xcomb[:,0]))))
                Xcombi[:,d],Xcombj[:,d] = Xcomb[:,0],Xcomb[:,1]
            for p in range(P):
                Ycomb = torch.combinations(Y[:,p])
                if not self.asymmetric_pairs:
                    Ycomb = torch.column_stack((torch.row_stack((Ycomb[:,0],Ycomb[:,1])),
                                                torch.row_stack((Ycomb[:,1],Ycomb[:,0]))))
                Ycombi[:,p],Ycombj[:,p] = Ycomb[:,0],Ycomb[:,1]  
            
            Inputs,Outputs = torch.column_stack((Xcombi,Xcombj,Ycombj)),Ycombi
            
        return Inputs,Outputs

