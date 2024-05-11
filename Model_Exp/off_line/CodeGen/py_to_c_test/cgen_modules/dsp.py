from string import Template

def arm_add_q7(pSrcA, pSrcB, pDst, blockSize):
	"""
	* @brief Q7 vector addition.    
	* @param[in]       *pSrcA points to the first input vector    
	* @param[in]       *pSrcB points to the second input vector    
	* @param[out]      *pDst points to the output vector    
	* @param[in]       blockSize number of samples in each vector    
	* @return none.    

	* Scaling and Overflow Behavior:
	* The function uses saturating arithmetic.    
	* Results outside of the allowable Q7 range [0x80 0x7F] will be saturated.    
	"""
 
	return Template("""
		arm_add_q7(${pSrcA}, ${pSrcB}, ${pDst}, ${blockSize});""").substitute(pSrcA=pSrcA, 
																	pSrcB=pSrcB, 
																	pDst=pDst, 
																	blockSize=blockSize)
  