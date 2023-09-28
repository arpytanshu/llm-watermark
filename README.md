
Minimal Implementation of  [``A watermark for Large Language Models``](https://arxiv.org/pdf/2301.10226.pdf)



### Algorithm 2: Text Generation with Soft Red List

Input:
- $s^{(−Np)} \ldots s^{-1}$ prompt of length $N_p$
- $\gamma \in(0,1)$ `green list size`
- $\delta > 0$  `hardness parameter`

Output:  
- $s^{(0)} \ldots s^{(T)}$ token generated by language model.

**for** $ t = 0,1, \ldots $ **do**
1. Apply the language model to prior tokens $ s^{(−Np)} \ldots s^{(t-1)}$  to get a logit vector $l^{(t)} $ over the vocabulary.
2. Compute a hash of token $s^{(t-1)}$.  
Use it to seed a RNG.
3. Using this RNG, randomly partition the vocabulary into:  
a **green list** $G$ of size $ \gamma |V| $ and  
a **red list** $R$ of size $ (1−\gamma)|V| $
4. Add $\delta$ to each green list logit.   
Apply the softmax operator to these modified logits to get a probability distribution over the vocabulary.  
$\hat{p}_{k}^{(t)} = 
\left\{ 
    \begin{array} {rcl}
    \frac
        { \exp ( l^{(t)}_k  + \delta)}
        {\sum_{i\in R} \exp (l^{(t)}_i) + \sum_{i\in G} \exp ( l^{(t)}_i + \delta)},
    & k \in G \\
    \frac
        { \exp ( l^{(t)}_k)}
        {\sum_{i\in R} \exp (l^{(t)}_i) + \sum_{i\in G} \exp ( l^{(t)}_i + \delta)},
    & k \in R
   \end{array} 
\right\}$
5. Sample the next token, $s^{(t)}$ , using the water-marked distribution $\hat{p}^{(t)}$.


#### sample out

ToDo:
1. create comparison charts portraying effect of hardness param vs number of tokens needed for detection.
2. create comparison charts portraying number of tokens needed for detection for hard and soft modes.

