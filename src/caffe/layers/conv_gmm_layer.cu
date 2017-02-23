#include <vector>
#include "caffe/layers/conv_gmm_layer.hpp"
#include <ctime>

using namespace std;
namespace caffe {

template <typename Dtype>
void ConvolutionGMMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
	
	for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
	weight = this->blobs_[0]->cpu_data();

	const Dtype* means = means_.cpu_data();
  const Dtype* invcovs = invcovs_.cpu_data();
  const Dtype* priors = priors_.cpu_data();
  
  Dtype * temp_data = temp_.mutable_cpu_data();
  Dtype * xc_data = xcentered_.mutable_cpu_data();
  Dtype temp = 0 ;
 	Dtype dot, sloss = -100000000;
	Dtype reg_loss = 0 ;  
	//for each kernel calculate membership and pick max 
  //for each 5th pass re-calculate membership 
	if(this->phase_ == 0){ // if not test
	
		for(int i = 0 ; i< this->blobs_[0]->channels()* this->blobs_[0]->num();++i){
	  	temp = 0 ;
	  	sloss = -100000000;	 
		// compute w - membership confidences
		// clock_t begin = clock();
 		 	for (int j=0; j<K_; j++ ) {  
			// This loop takes 0.05 sec in caffe_gpu op and 5.e-5 on cpu mode

      // take mean out
      caffe_sub(N_, weight +i*N_, means + j*N_, xc_data);        
      
      // apply inverse covariance 
      caffe_mul(N_, xc_data, invcovs + j*N_, temp_data);    			
			
			dot = caffe_cpu_dot(N_, temp_data, xc_data);     
      
      // compute the membership confidences
      temp  =  - 0.5 * dot;
      
      // log-sum-exp approximation - pick the best component
      if ( sloss < temp + priors[j] )  {
					sloss = temp + priors[j];
					activations_[i] = j ;
        }             
      }

		 	reg_loss =+ sloss;
	//	 clock_t end = clock();
	//	 double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		}
	
pass_+=1;
//		LOG(INFO) << "reg_loss :" << reg_loss << " " ;
	}	
}

template <typename Dtype>
void ConvolutionGMMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
 	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
 	
  const Dtype reg_factor = reg_factor_;
	const Dtype* means = means_.cpu_data();
  const Dtype* invcovs = invcovs_.cpu_data();
  Dtype * temp_data = temp_.mutable_cpu_data();
  Dtype * xc_data = xcentered_.mutable_cpu_data();

	// Add regularization loss
	for(int j = 0 ; j< this->blobs_[0]->channels()* this->blobs_[0]->num();++j){
		 //calculate log-derivative 
		caffe_sub(N_, weight + j*N_, means +activations_[j]*N_, xc_data);        
		caffe_mul(N_, xc_data, invcovs + activations_[j]*N_,temp_data);
			
		// add GMM-derivative to Classification-derivative
		caffe_axpy(N_,reg_factor,temp_data,weight_diff +j*N_);
  	//  LOG(INFO) << activations_[i] << " " ;
	}	
		
	weight_diff = this->blobs_[0]->mutable_gpu_diff();	
	for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
				}
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionGMMLayer);

}  // namespace caffe
