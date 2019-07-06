#include <cstring>
#include <immintrin.h>
#include <emmintrin.h>

struct InputTensor_t {
    __m256 val;
    int index;
};
typedef struct InputTensor_t InputTensor_t;

struct input_t {
    float val;
    uint16_t index;
};
typedef struct input_t input_t;

void transposeMatrix( float*a, float* aT, int m, int n ) {
	for( auto i=0; i<m; ++i ) {
		for( auto j=0; j<n; ++j ) {
			aT[j*m+i]=a[i*n+j];
		}
	}
}

void transposeMatrix( float*a, uint16_t* aT, int m, int n, int bit_a ) {
	float fp_factor = float(pow(2,bit_a)-1);
	for( auto i=0; i<m; ++i ) {
		for( auto j=0; j<n; ++j ) {
			aT[j*m+i] = (uint16_t) round( a[i*n+j] * fp_factor );
		}
	}
}

void createSparseList( float* a, uint16_t**& ap, float Wp, float Wn, int k, int n ) {

	ap = (uint16_t**)malloc( 2*n * sizeof(uint16_t*));

	int* ap_tmp = (int*)malloc( k * sizeof(int));
	int* an_tmp = (int*)malloc( k * sizeof(int));
  int ap_cnt=0, an_cnt=0;
  int neg_cnt = 0;
  float val;
  for( auto i=0; i<n; ++i ) {
		ap_cnt=0;
		an_cnt=0;
    for( auto j=0; j<k; ++j ) {
    	val = a[i*k+j];
      if( val == Wp ) {
      	ap_cnt++;
        ap_tmp[ap_cnt]=j;
      }
      else if( val == Wn ) {
      	an_cnt++;
        an_tmp[an_cnt]=j;
      }
      else if( val == 0 ) {
      	neg_cnt++;
      }
      else {
      	std::cout << "[ERROR] Value is out of range" << std::endl;
      }
    }

		ap[i*2+0] = (uint16_t*)malloc((1 + ap_cnt) * sizeof(uint16_t));
		ap[i*2+1] = (uint16_t*)malloc((1 + an_cnt) * sizeof(uint16_t));

		ap[i*2+0][0] = ap_cnt;
		ap[i*2+1][0] = an_cnt;

		for( auto col=1; col<=ap_cnt; ++col ) {
    	ap[i*2+0][col] = (uint16_t) ap_tmp[col];
    }
		for( auto col=1; col<=an_cnt; ++col ) {
    	ap[i*2+1][col] = (uint16_t) an_tmp[col];
    }
  }
  free( ap_tmp );
  free( an_tmp );
}

typedef struct {
  int8_t select;
  int16_t index;
} sparse_element_t;

typedef struct {
  int elements;
  sparse_element_t* list;
} sparse_list_t;

sparse_list_t* createSparseListv2( float* a, float Wp, float Wn, int k, int n ) {

  sparse_list_t* weight_list = (sparse_list_t*) malloc( n * sizeof(sparse_list_t) );

  sparse_element_t* list_tmp = (sparse_element_t*) malloc( k * sizeof(sparse_element_t) );

  int last_index=0;
  int neg_cnt = 0;
  float val;
  for( int i=0; i<n; ++i ) {
    last_index=0;
    for( int j=0; j<k; ++j ) {
    	val = a[i*k+j];
      if( val == Wp ) {
        list_tmp[last_index].select = 0;
        list_tmp[last_index].index = j;
        last_index++;
      }
      else if( val == Wn ) {
        list_tmp[last_index].select = 1;
        list_tmp[last_index].index = j;
        last_index++;
      }
      else if( val == 0 ) {
      	neg_cnt++;
      }
      else {
      	std::cout << "[ERROR] Value is out of range" << std::endl;
      }
    }
    (weight_list[i].list) = (sparse_element_t*) malloc( last_index * sizeof(sparse_element_t*) );
    weight_list[i].elements = last_index;
    if(last_index>1024) std::cout << "last index is zero";
    for( auto el=0; el<last_index; ++el ) {
      weight_list[i].list[el].select = list_tmp[el].select;
      weight_list[i].list[el].index = list_tmp[el].index;
    }
  }
  free( list_tmp );

  return weight_list;
}
/*
void SparseDotProductnew( float* a, sparse_list_t* weight_list, float* c, float Wp, float Wn, int m, int n, int k ) {
	__m256 w_p = _mm256_set1_ps( Wp );
	__m256 w_n = _mm256_set1_ps( Wn );
	__m256 accu_p;
	__m256 accu_n;
  __m256 scale;

  for( int row=0; row<k; ++row ) {
    int elements = weight_list[row].elements;
    for( int el=0; el<elements; ++el ) {
      int index = weight_list[row].list[el].index;
      int select = weight_list[row].list[el].select;
      if( select==0 ) {
        scale = w_p;
      }
      else if( select==1 ) {
        scale = w_n;
      }
      std::cout << "Select = " << select << std::endl;
      for( int input_row=0; input_row<m; input_row+=8 ) {
        int in_element= input_row+m*index;
  			__m256 input_val = _mm256_load_ps( &a[in_element] );
        __m256 res = _mm256_load_ps( &c[input_row + m*row] );
        //res = _mm256_add_ps( _mm256_mul_ps ( input_val, scale ), res );
        res = _mm256_fmadd_ps( input_val, scale, res);
        _mm256_storeu_ps( &c[input_row + m*row], res );
      }
    }
  }
}*/

void SparseDotProduct( float* a, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k ) {
  float *c_p = (float*)malloc( m * sizeof(float));
  float *c_n = (float*)malloc( m * sizeof(float));

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(float));
    std::memset(c_n, 0, m*sizeof(float));

    //int8x16_t accu_p4 = vdupq_n_s8(0);//vcreate_f32(0);
    //int8x16_t accu_n4 = vdupq_n_s8(0);//= vcreate_f32(0);
		float accu_p = 0.0;
		float accu_n = 0.0;

		int num_cols_p = (int)b[row*2+0][0];
		int num_cols_n = (int)b[row*2+1][0];

		bool done_p=false, done_n=false;
		int col_p=0, col_n=0;

    //for( int col=1; col<=num_cols_p; ++col ) {
		do {
			int index_p = (int)b[row*2+0][col_p];
			int index_n = (int)b[row*2+1][col_n];

			if( done_p==false && (index_p<index_n || done_n==true) )
			//if( done_p==false )
			{
			for( int input_row=0; input_row<m; ++input_row ) {
				float input_val = a[input_row+m*index_p];
				//if( input_val!=0 )
				{
				accu_p = c_p[input_row];
				accu_p = accu_p + input_val;
				c_p[input_row] = accu_p;
				}
      }
			if(col_p<num_cols_p) col_p++;
			else done_p=true;
			}


			//if( done_n==false && (index_p>index_n || done_p==true) )
			else if( done_n==false )
			{
			for( int input_row=0; input_row<m; ++input_row ) {
				float input_val = a[input_row+m*index_n];
				//if(input_val!=0)
				{
				accu_n = c_n[input_row];
				accu_n = accu_n + input_val;
				c_n[input_row] = accu_n;
				}
      }
			if(col_n<num_cols_n) col_n++;
			else done_n=true;
			}

    } while(done_p!=true || done_n!=true);

		/*int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
    	for( int input_row=0; input_row<m; ++input_row ) {
				int index = input_row+m*(int)b[row*2+1][col];
				float input_val = a[index];
				//if(input_val!=0)
				{
				accu_n = c_n[input_row];
				accu_n = accu_n + input_val;
				c_n[input_row] = accu_n;
				}
      }
    }*/

    for( int input_row=0; input_row<m; ++input_row ) {
			accu_p = c_p[input_row];
			accu_n = c_n[input_row];
			float res = accu_p * Wp + accu_n * Wn;
			c[input_row + m*row] = res;
    }
  }
}

void SASparseDotProduct( input_t* a, uint16_t* indices, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k ) {
  float *c_p = (float*)malloc( m * sizeof(float));
  float *c_n = (float*)malloc( m * sizeof(float));

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(float));
    std::memset(c_n, 0, m*sizeof(float));

    //int8x16_t accu_p4 = vdupq_n_s8(0);//vcreate_f32(0);
    //int8x16_t accu_n4 = vdupq_n_s8(0);//= vcreate_f32(0);
		float accu_p = 0.0;
		float accu_n = 0.0;

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
			int offset = m*(int)b[row*2+0][col];
			//int a_cols=indices[(m+1)*(int)b[row*2+0][col]];
			int a_cols=a[(m+2)*(int)b[row*2+0][col]].val;
			//std::cout << a_cols << "/" << m << std::endl;
			for( int input_row=0; input_row<a_cols; ++input_row ) {
				//accu_p = c_p[input_row];
				//accu_p = c_p[indices[input_row+1]];
				accu_p = c_p[a[input_row+1].index];
				//int index = input_row+m*(int)b[row*2+0][col];
				int index = input_row+offset;
				float input_val = a[index+1].val;
				accu_p = accu_p + input_val;
				c_p[a[input_row+1].index] = accu_p;
      }
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
			int offset = m*(int)b[row*2+1][col];
			//int a_cols=indices[(m+1)*(int)b[row*2+1][col]];
			int a_cols=a[(m+2)*(int)b[row*2+1][col]].val;
    	for( int input_row=0; input_row<a_cols; ++input_row ) {
				//accu_n = c_n[input_row];
				accu_n = c_n[a[input_row+1].index];
				int index = input_row+offset;
				float input_val = a[index+1].val;
				accu_n = accu_n + input_val;
				//c_n[input_row] = accu_n;
				c_n[a[input_row+1].index]=accu_n;
      }
    }

    for( int input_row=0; input_row<m; ++input_row ) {
			accu_p = c_p[input_row];
			accu_n = c_n[input_row];
			float res = accu_p * Wp + accu_n * Wn;
			c[input_row + m*row] = res;
    }
  }
}

void FcBnReLU( uint8_t* a, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k, int bit_a ) {
  int *c_p = (int*)malloc( m * sizeof(int));
  int *c_n = (int*)malloc( m * sizeof(int));

	float fp_factor =  float(pow(2, bit_a)-1);

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(int));
    std::memset(c_n, 0, m*sizeof(int));

    //int8x16_t accu_p4 = vdupq_n_s8(0);//vcreate_f32(0);
    //int8x16_t accu_n4 = vdupq_n_s8(0);//= vcreate_f32(0);
		int accu_p = 0.0;
		int accu_n = 0.0;

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
			for( int input_row=0; input_row<m; ++input_row ) {
				accu_p = c_p[input_row];
				int index = input_row+m*(int)b[row*2+0][col];
				int input_val = (int) a[index];
				if(input_val>255 || input_val < 0) std::cout << "out of range" << std::endl;
				accu_p = accu_p + input_val;
				c_p[input_row] = accu_p;
      }
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
    	for( int input_row=0; input_row<m; ++input_row ) {
				accu_n = c_n[input_row];
				int index = input_row+m*(int)b[row*2+1][col];
				int input_val = (int) a[index];
				if(input_val>255 || input_val < 0) std::cout << "out of range" << std::endl;
				accu_n = accu_n + input_val;
				c_n[input_row] = accu_n;
      }
    }

    for( int input_row=0; input_row<m; ++input_row ) {
			float res_p = (float) c_p[input_row];
			float res_n = (float) c_n[input_row];
			res_p = res_p / fp_factor;
			res_n = res_n / fp_factor;
			float res = res_p * Wp + res_n * Wn;
			c[input_row + m*row] = res;
    }
  }
}

#ifdef USE_AVX
void FcBnReLUAVX2( uint16_t* a, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k, int bit_a ) {
  uint16_t *c_p = (uint16_t*)malloc( m * sizeof(uint16_t));
	//float *accu_p_sp = (float*)malloc( m * sizeof(float));
  uint16_t *c_n = (uint16_t*)malloc( m * sizeof(uint16_t));

	float fp_factor =  float(pow(2, bit_a)-1);

	__m256 w_p = _mm256_set1_ps( Wp/fp_factor );
	__m256 w_n = _mm256_set1_ps( Wn/fp_factor );
	//__m256 w_p = _mm256_set1_ps( 1);
	//__m256 w_n = _mm256_set1_ps( 1 );
	//__m256 w_p = _mm256_set1_ps( Wp );
	//__m256 w_n = _mm256_set1_ps( Wn );

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(uint16_t));
    std::memset(c_n, 0, m*sizeof(int));

    //int8x16_t accu_p4 = vdupq_n_s8(0);//vcreate_f32(0);
    //int8x16_t accu_n4 = vdupq_n_s8(0);//= vcreate_f32(0);
		//float accu_p = 0.0;
		//float accu_n = 0.0;

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
			for( int input_row=0; input_row<m; input_row+=16 ) {
				//accu_p = c_p[input_row];
				int index = input_row+m*(int)b[row*2+0][col];
				//int input_val = (int) a[index];
				__m256i input8bit = _mm256_load_si256( (__m256i*) &a[index] );
				/*
				__m256i input8bit = _mm256_set_epi16 (	a[index+15], \
																								a[index+14], \
																								a[index+13], \
																								a[index+12], \
																								a[index+11], \
																								a[index+10], \
																								a[index+9], \
																								a[index+8], \
																								a[index+7], \
																								a[index+6], \
																								a[index+5], \
																								a[index+4], \
																								a[index+3], \
																								a[index+2], \
																								a[index+1], \
																								a[index+0] );*/

				/*__m256i input_low = _mm256_cvtepu8_epi16( _mm256_castsi256_si128(input8bit) );
				//__m256i input_low = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 0) );
				__m256i accu_p_low = _mm256_load_si256( (__m256i*) &c_p[input_row] );

				accu_p_low = _mm256_add_epi16( accu_p_low, input_low );
				_mm256_store_si256( (__m256i*) &c_p[input_row+0], accu_p_low );

				__m256i input_high = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 1) );
				__m256i accu_p_high = _mm256_load_si256( (__m256i*) &c_p[input_row+16] );

				accu_p_high = _mm256_add_epi16( accu_p_high, input_high );
				_mm256_store_si256( (__m256i*) &c_p[input_row+16], accu_p_high );*/
				//__m256i accu = _mm256_load_si256( (__m256i*) &c_p[input_row] );
				__m256i accu = _mm256_set_epi16 (				c_p[input_row+15], \
																								c_p[input_row+14], \
																								c_p[input_row+13], \
																								c_p[input_row+12], \
																								c_p[input_row+11], \
																								c_p[input_row+10], \
																								c_p[input_row+9], \
																								c_p[input_row+8], \
																								c_p[input_row+7], \
																								c_p[input_row+6], \
																								c_p[input_row+5], \
																								c_p[input_row+4], \
																								c_p[input_row+3], \
																								c_p[input_row+2], \
																								c_p[input_row+1], \
																								c_p[input_row+0] );

				accu = _mm256_add_epi16( accu, input8bit );
				_mm256_store_si256( (__m256i*) &c_p[input_row], accu );
      }
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
    	for( int input_row=0; input_row<m; input_row+=16 ) {
				int index = input_row+m*(int)b[row*2+1][col];
				__m256i input8bit = _mm256_load_si256( (__m256i*) &a[index] );
				/*__m256i input8bit = _mm256_set_epi16 (	a[index+15], \
																								a[index+14], \
																								a[index+13], \
																								a[index+12], \
																								a[index+11], \
																								a[index+10], \
																								a[index+9], \
																								a[index+8], \
																								a[index+7], \
																								a[index+6], \
																								a[index+5], \
																								a[index+4], \
																								a[index+3], \
																								a[index+2], \
																								a[index+1], \
																								a[index+0] );*/
				/*
				__m256i input_low = _mm256_cvtepu8_epi16( _mm256_castsi256_si128(input8bit) );
				//__m256i input_low = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 0) );
				__m256i accu_n_low = _mm256_load_si256( (__m256i*) &c_n[input_row] );
				accu_n_low = _mm256_add_epi16( accu_n_low, input_low );
				_mm256_store_si256( (__m256i*) &c_n[input_row+0], accu_n_low );

				__m256i input_high = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 1) );
				__m256i accu_n_high = _mm256_load_si256( (__m256i*) &c_n[input_row+16] );

				accu_n_high = _mm256_add_epi16( accu_n_high, input_high );
				_mm256_store_si256( (__m256i*) &c_n[input_row+16], accu_n_high );*/
				//__m256i accu = _mm256_load_si256( (__m256i*) &c_n[input_row] );
				__m256i accu = _mm256_set_epi16 (				c_n[input_row+15], \
																								c_n[input_row+14], \
																								c_n[input_row+13], \
																								c_n[input_row+12], \
																								c_n[input_row+11], \
																								c_n[input_row+10], \
																								c_n[input_row+9], \
																								c_n[input_row+8], \
																								c_n[input_row+7], \
																								c_n[input_row+6], \
																								c_n[input_row+5], \
																								c_n[input_row+4], \
																								c_n[input_row+3], \
																								c_n[input_row+2], \
																								c_n[input_row+1], \
																								c_n[input_row+0] );

				accu = _mm256_add_epi16( accu, input8bit );
				_mm256_store_si256( (__m256i*) &c_n[input_row], accu );
      }
    }

    /*for( int input_row=0; input_row<m; ++input_row ) {
			float res_p = (float) c_p[input_row];
			float res_n = (float) c_n[input_row];
			res_p = res_p / fp_factor;
			res_n = res_n / fp_factor;
			float res = res_p * Wp + res_n * Wn;
			c[input_row + m*row] = res;
    }*/

		for( int input_row=0; input_row<m; input_row+=8 ) {
			//__m256 accu_p = _mm256_castsi256_ps( _mm256_cvtepu16_epi32( _mm256_castsi256_si128( _mm256_load_si256( (__m256i*) &c_p[input_row] ) ) ) );
			//__m256 accu_n = _mm256_castsi256_ps( _mm256_cvtepu16_epi32( _mm256_castsi256_si128( _mm256_load_si256( (__m256i*) &c_n[input_row] ) ) ) );
			//accu_n = _mm256_load_ps( &c_n[input_row] );
			__m256 accu_p = _mm256_set_ps ( (float) c_p[input_row+7], \
			 																(float) c_p[input_row+6], \
																			(float) c_p[input_row+5], \
																			(float) c_p[input_row+4], \
																			(float) c_p[input_row+3], \
																			(float) c_p[input_row+2], \
																			(float) c_p[input_row+1], \
																			(float) c_p[input_row+0] );

			__m256 accu_n = _mm256_set_ps ( (float) c_n[input_row+7], \
																			(float) c_n[input_row+6], \
																			(float) c_n[input_row+5], \
																			(float) c_n[input_row+4], \
																			(float) c_n[input_row+3], \
																			(float) c_n[input_row+2], \
																			(float) c_n[input_row+1], \
																			(float) c_n[input_row+0] );

			__m256 res_p = _mm256_mul_ps( accu_p, w_p );
			__m256 res_n = _mm256_mul_ps( accu_n, w_n );
			__m256 res = _mm256_add_ps( res_p, res_n );
			_mm256_storeu_ps( &c[input_row + m*row], res );
			//std::cout << c[input_row + m*row] << std::endl;
    }
  }
	//free(c_p);
	//free(c_n);
}

void SAFcBnReLUAVX2( uint16_t* a, uint16_t* indices, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k, int bit_a ) {
  uint16_t *c_p = (uint16_t*)malloc( m * sizeof(uint16_t));
	//float *accu_p_sp = (float*)malloc( m * sizeof(float));
  uint16_t *c_n = (uint16_t*)malloc( m * sizeof(uint16_t));

	float fp_factor =  float(pow(2, bit_a)-1);

	__m256 w_p = _mm256_set1_ps( Wp/fp_factor );
	__m256 w_n = _mm256_set1_ps( Wn/fp_factor );
	//__m256 w_p = _mm256_set1_ps( 1);
	//__m256 w_n = _mm256_set1_ps( 1 );
	//__m256 w_p = _mm256_set1_ps( Wp );
	//__m256 w_n = _mm256_set1_ps( Wn );

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(uint16_t));
    std::memset(c_n, 0, m*sizeof(int));

    //int8x16_t accu_p4 = vdupq_n_s8(0);//vcreate_f32(0);
    //int8x16_t accu_n4 = vdupq_n_s8(0);//= vcreate_f32(0);
		//float accu_p = 0.0;
		//float accu_n = 0.0;

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
			int offset = m*(int)b[row*2+0][col];
			int a_cols=indices[(m+1)*(int)b[row*2+0][col]];
			//std::cout << "a_cols=" << a_cols << std::endl;
			//assert( a_cols==m );
			int a_cols_vec=a_cols/16;
			int a_cols_seq=a_cols%16;
			for( int input_row=0; input_row<a_cols_vec; input_row+=16 ) {
				//accu_p = c_p[input_row];
				int index = input_row+offset;
				//int input_val = (int) a[index];
				__m256i input8bit = _mm256_load_si256( (__m256i*) &a[index] );
				/*
				__m256i input8bit = _mm256_set_epi16 (	a[index+15], \
																								a[index+14], \
																								a[index+13], \
																								a[index+12], \
																								a[index+11], \
																								a[index+10], \
																								a[index+9], \
																								a[index+8], \
																								a[index+7], \
																								a[index+6], \
																								a[index+5], \
																								a[index+4], \
																								a[index+3], \
																								a[index+2], \
																								a[index+1], \
																								a[index+0] );*/

				/*__m256i input_low = _mm256_cvtepu8_epi16( _mm256_castsi256_si128(input8bit) );
				//__m256i input_low = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 0) );
				__m256i accu_p_low = _mm256_load_si256( (__m256i*) &c_p[input_row] );

				accu_p_low = _mm256_add_epi16( accu_p_low, input_low );
				_mm256_store_si256( (__m256i*) &c_p[input_row+0], accu_p_low );

				__m256i input_high = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 1) );
				__m256i accu_p_high = _mm256_load_si256( (__m256i*) &c_p[input_row+16] );

				accu_p_high = _mm256_add_epi16( accu_p_high, input_high );
				_mm256_store_si256( (__m256i*) &c_p[input_row+16], accu_p_high );*/
				//__m256i accu = _mm256_load_si256( (__m256i*) &c_p[input_row] );
				__m256i accu = _mm256_set_epi16 (				c_p[indices[input_row+15+1]], \
																								c_p[indices[input_row+14+1]], \
																								c_p[indices[input_row+13+1]], \
																								c_p[indices[input_row+12+1]], \
																								c_p[indices[input_row+11+1]], \
																								c_p[indices[input_row+10+1]], \
																								c_p[indices[input_row+9+1]], \
																								c_p[indices[input_row+8+1]], \
																								c_p[indices[input_row+7+1]], \
																								c_p[indices[input_row+6+1]], \
																								c_p[indices[input_row+5+1]], \
																								c_p[indices[input_row+4+1]], \
																								c_p[indices[input_row+3+1]], \
																								c_p[indices[input_row+2+1]], \
																								c_p[indices[input_row+1+1]], \
																								c_p[indices[input_row+0+1]] );

				accu = _mm256_add_epi16( accu, input8bit );
				//_mm256_store_si256( (__m256i*) &c_p[input_row], accu );
				for( int vec_id=0; vec_id<16; ++vec_id ) {
				 	c_p[indices[input_row+vec_id+1]] = _mm256_extract_epi16( accu, vec_id );
				}
      }
			for( int input_row=a_cols_vec; input_row<a_cols; ++input_row ) {
				int index = input_row+offset;
			  int input_val = (int) a[index];
				c_p[indices[input_row+1]]+=input_val;
			}
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
    	for( int input_row=0; input_row<m; input_row+=16 ) {
				int index = input_row+m*(int)b[row*2+1][col];
				__m256i input8bit = _mm256_load_si256( (__m256i*) &a[index] );
				/*__m256i input8bit = _mm256_set_epi16 (	a[index+15], \
																								a[index+14], \
																								a[index+13], \
																								a[index+12], \
																								a[index+11], \
																								a[index+10], \
																								a[index+9], \
																								a[index+8], \
																								a[index+7], \
																								a[index+6], \
																								a[index+5], \
																								a[index+4], \
																								a[index+3], \
																								a[index+2], \
																								a[index+1], \
																								a[index+0] );*/
				/*
				__m256i input_low = _mm256_cvtepu8_epi16( _mm256_castsi256_si128(input8bit) );
				//__m256i input_low = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 0) );
				__m256i accu_n_low = _mm256_load_si256( (__m256i*) &c_n[input_row] );
				accu_n_low = _mm256_add_epi16( accu_n_low, input_low );
				_mm256_store_si256( (__m256i*) &c_n[input_row+0], accu_n_low );

				__m256i input_high = _mm256_cvtepu8_epi16( _mm256_extracti128_si256(input8bit, 1) );
				__m256i accu_n_high = _mm256_load_si256( (__m256i*) &c_n[input_row+16] );

				accu_n_high = _mm256_add_epi16( accu_n_high, input_high );
				_mm256_store_si256( (__m256i*) &c_n[input_row+16], accu_n_high );*/
				//__m256i accu = _mm256_load_si256( (__m256i*) &c_n[input_row] );
				__m256i accu = _mm256_set_epi16 (				c_n[input_row+15], \
																								c_n[input_row+14], \
																								c_n[input_row+13], \
																								c_n[input_row+12], \
																								c_n[input_row+11], \
																								c_n[input_row+10], \
																								c_n[input_row+9], \
																								c_n[input_row+8], \
																								c_n[input_row+7], \
																								c_n[input_row+6], \
																								c_n[input_row+5], \
																								c_n[input_row+4], \
																								c_n[input_row+3], \
																								c_n[input_row+2], \
																								c_n[input_row+1], \
																								c_n[input_row+0] );

				accu = _mm256_add_epi16( accu, input8bit );
				_mm256_store_si256( (__m256i*) &c_n[input_row], accu );
      }
    }

    /*for( int input_row=0; input_row<m; ++input_row ) {
			float res_p = (float) c_p[input_row];
			float res_n = (float) c_n[input_row];
			res_p = res_p / fp_factor;
			res_n = res_n / fp_factor;
			float res = res_p * Wp + res_n * Wn;
			c[input_row + m*row] = res;
    }*/

		for( int input_row=0; input_row<m; input_row+=8 ) {
			//__m256 accu_p = _mm256_castsi256_ps( _mm256_cvtepu16_epi32( _mm256_castsi256_si128( _mm256_load_si256( (__m256i*) &c_p[input_row] ) ) ) );
			//__m256 accu_n = _mm256_castsi256_ps( _mm256_cvtepu16_epi32( _mm256_castsi256_si128( _mm256_load_si256( (__m256i*) &c_n[input_row] ) ) ) );
			//accu_n = _mm256_load_ps( &c_n[input_row] );
			__m256 accu_p = _mm256_set_ps ( (float) c_p[input_row+7], \
			 																(float) c_p[input_row+6], \
																			(float) c_p[input_row+5], \
																			(float) c_p[input_row+4], \
																			(float) c_p[input_row+3], \
																			(float) c_p[input_row+2], \
																			(float) c_p[input_row+1], \
																			(float) c_p[input_row+0] );

			__m256 accu_n = _mm256_set_ps ( (float) c_n[input_row+7], \
																			(float) c_n[input_row+6], \
																			(float) c_n[input_row+5], \
																			(float) c_n[input_row+4], \
																			(float) c_n[input_row+3], \
																			(float) c_n[input_row+2], \
																			(float) c_n[input_row+1], \
																			(float) c_n[input_row+0] );

			__m256 res_p = _mm256_mul_ps( accu_p, w_p );
			__m256 res_n = _mm256_mul_ps( accu_n, w_n );
			__m256 res = _mm256_add_ps( res_p, res_n );
			_mm256_storeu_ps( &c[input_row + m*row], res );
			//std::cout << c[input_row + m*row] << std::endl;
    }
  }
	//free(c_p);
	//free(c_n);
}
/*
void SASparseDotProductAVX2( InputTensor_t* a, int* size_vec, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k ) {
  float*c_p = (float*)malloc( (m+8) * sizeof(float));
  float *c_n = (float*)malloc( (m+8) * sizeof(float));

	__m256 w_p = _mm256_set1_ps( Wp );
	__m256 w_n = _mm256_set1_ps( Wn );
	__m256 accu_p;
	__m256 accu_n;

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(float));
    std::memset(c_n, 0, m*sizeof(float));

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
      int a_row=(int)b[row*2+0][col];
      int a_elements = size_vec[a_row];
			for( int input_row=0; input_row<a_elements; ++input_row ) {
        int index = input_row+m*a_row;
        int accu_index = a[inxex].index;
        __m256 input_val = = a[inxex].val;
				accu_p = _mm256_load_ps( &c_p[accu_index] );
				//int index = input_row+m*(int)b[row*2+0][col];
				//__m256 input_val = _mm256_load_ps( &a[index] );
				accu_p = _mm256_add_ps( accu_p, input_val );
				_mm256_storeu_ps( &c_p[accu_index], accu_p );
			}
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
			for( int input_row=0; input_row<m; input_row+=8 ) {
				accu_n = _mm256_load_ps( &c_n[input_row] );
				int index = input_row+m*(int)b[row*2+1][col];
				__m256 input_val = _mm256_load_ps( &a[index] );
				accu_n = _mm256_add_ps( accu_n, input_val );
				_mm256_storeu_ps( &c_n[input_row], accu_n );
      }
    }

    for( int input_row=0; input_row<m; input_row+=8 ) {
			accu_p = _mm256_load_ps( &c_p[input_row] );
			accu_n = _mm256_load_ps( &c_n[input_row] );
			__m256 res_p = _mm256_mul_ps( accu_p, w_p );
			__m256 res_n = _mm256_mul_ps( accu_n, w_n );
			__m256 res = _mm256_add_ps( res_p, res_n );
			_mm256_storeu_ps( &c[input_row + m*row], res );
    }
  }
	free(c_n);
	free(c_p);
}
*/
void SparseDotProductAVX2( float* a, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k ) {
  float *c_p = (float*)malloc( m * sizeof(float));
  float *c_n = (float*)malloc( m * sizeof(float));

	__m256 w_p = _mm256_set1_ps( Wp );
	__m256 w_n = _mm256_set1_ps( Wn );
	__m256 accu_p;
	__m256 accu_n;

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(float));
    std::memset(c_n, 0, m*sizeof(float));

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
      int glob_index = (int)b[row*2+0][col];
			for( int input_row=0; input_row<m; input_row+=8 ) {
				accu_p = _mm256_load_ps( &c_p[input_row] );
				int index = input_row+m*glob_index;
				__m256 input_val = _mm256_load_ps( &a[index] );
				accu_p = _mm256_add_ps( accu_p, input_val );
				_mm256_storeu_ps( &c_p[input_row], accu_p );
			}
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
      int glob_index = (int)b[row*2+1][col];
			for( int input_row=0; input_row<m; input_row+=8 ) {
				accu_n = _mm256_load_ps( &c_n[input_row] );
				int index = input_row+m*glob_index;
				__m256 input_val = _mm256_load_ps( &a[index] );
				accu_n = _mm256_add_ps( accu_n, input_val );
				_mm256_storeu_ps( &c_n[input_row], accu_n );
      }
    }

    for( int input_row=0; input_row<m; input_row+=8 ) {
			accu_p = _mm256_load_ps( &c_p[input_row] );
			accu_n = _mm256_load_ps( &c_n[input_row] );
			__m256 res_p = _mm256_mul_ps( accu_p, w_p );
			__m256 res_n = _mm256_mul_ps( accu_n, w_n );
			__m256 res = _mm256_add_ps( res_p, res_n );
			_mm256_storeu_ps( &c[input_row + m*row], res );
    }
  }
	free(c_n);
	free(c_p);
}

void SparseDotProductAVX2tmp( float* a, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k ) {
  float *c_p = (float*)malloc( m * sizeof(float));
  float *c_n = (float*)malloc( m * sizeof(float));

	__m256 w_p = _mm256_set1_ps( Wp );
	__m256 w_n = _mm256_set1_ps( Wn );
	__m256 accu_p;
	__m256 accu_n;

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(float));
    std::memset(c_n, 0, m*sizeof(float));

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
      if(row<k-1) {
        int next_index = 0+m*(int)b[(row+1)*2+0][col];
        //_mm_prefetch(&a[next_index], 1);
      }
			for( int input_row=0; input_row<m; input_row+=8 ) {
				accu_p = _mm256_load_ps( &c_p[input_row] );
				int index = input_row+m*(int)b[row*2+0][col];
				__m256 input_val = _mm256_load_ps( &a[index] );
				accu_p = _mm256_add_ps( accu_p, input_val );
				_mm256_storeu_ps( &c_p[input_row], accu_p );
			}
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
			for( int input_row=0; input_row<m; input_row+=8 ) {
				accu_n = _mm256_load_ps( &c_n[input_row] );
				int index = input_row+m*(int)b[row*2+1][col];
				__m256 input_val = _mm256_load_ps( &a[index] );
				accu_n = _mm256_add_ps( accu_n, input_val );
				_mm256_storeu_ps( &c_n[input_row], accu_n );
      }
    }

    for( int input_row=0; input_row<m; input_row+=8 ) {
			accu_p = _mm256_load_ps( &c_p[input_row] );
			accu_n = _mm256_load_ps( &c_n[input_row] );
			__m256 res_p = _mm256_mul_ps( accu_p, w_p );
			__m256 res_n = _mm256_mul_ps( accu_n, w_n );
			__m256 res = _mm256_add_ps( res_p, res_n );
			_mm256_storeu_ps( &c[input_row + m*row], res );
    }
  }
	free(c_n);
	free(c_p);
}

void SparseDotProductAVX2old( float* a, uint16_t** b, float* c, float Wp, float Wn, int m, int n, int k ) {
  float*c_p = (float*)malloc( m * sizeof(float));
  float *c_n = (float*)malloc( m * sizeof(float));

	__m256 w_p = _mm256_set1_ps( Wp );
	__m256 w_n = _mm256_set1_ps( Wn );

  for( int row=0; row<k; ++row ) {
  	std::memset(c_p, 0, m*sizeof(float));
    std::memset(c_n, 0, m*sizeof(float));

    //int8x16_t accu_p4 = vdupq_n_s8(0);//vcreate_f32(0);
    //int8x16_t accu_n4 = vdupq_n_s8(0);//= vcreate_f32(0);
		//float accu_p2 = 0.0;
		__m256 accu_p;
		//float accu_n2 = 0.0;
		__m256 accu_n;

		int num_cols_p = (int)b[row*2+0][0];
    for( int col=1; col<=num_cols_p; ++col ) {
			for( int input_row=0; input_row<m; input_row+=8 ) {
				accu_p = _mm256_load_ps( &c_p[input_row] );
				int index = input_row+m*(int)b[row*2+0][col];
				__m256 input_val = _mm256_load_ps( &a[index] );
				accu_p = _mm256_add_ps( accu_p, input_val );
				_mm256_storeu_ps( &c_p[input_row], accu_p );
			}
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( int col=1; col<=num_cols_n; ++col ) {
    	for( int input_row=0; input_row<m; input_row+=8 ) {
				accu_n = _mm256_load_ps( &c_n[input_row] );
				int index = input_row+m*(int)b[row*2+1][col];
				//float input_val = a[index];
				__m256 input_val = _mm256_load_ps( &a[index] );
				//accu_p = accu_p + input_val;
				accu_n = _mm256_add_ps( accu_n, input_val );
				//c_p[input_row] = accu_p;
				_mm256_storeu_ps( &c_n[input_row], accu_n );
      }
    }

    for( int input_row=0; input_row<m; input_row+=8 ) {
			//accu_p2 = c_p[input_row];
			accu_p = _mm256_load_ps( &c_p[input_row] );
			//accu_n2 = c_n[input_row];
			accu_n = _mm256_load_ps( &c_n[input_row] );
			//float res = accu_p2 * Wp + accu_n2 * Wn;
			__m256 res_p = _mm256_mul_ps( accu_p, w_p );
			__m256 res_n = _mm256_mul_ps( accu_n, w_n );
			__m256 res = _mm256_add_ps( res_p, res_n );
			_mm256_storeu_ps( &c_n[input_row + m*row], res );
			//c[input_row + m*row] = res;
    }
  }
}
#endif

/*
void SparseDotProductNEON( int8_t* a, uint16_t** b, int8_t* c, int m, int n, int k ) {
#pragma omp parallel
{
	int id, Nthrds, istart, iend;
  id = omp_get_thread_num();
  Nthrds = omp_get_num_threads();
	istart = id * k / Nthrds;
  iend = (id+1) * k / Nthrds;
  if (id == Nthrds-1) iend = k;

  int8_t *c_p = (int8_t*)malloc( m * sizeof(int8_t));
  int8_t *c_n = (int8_t*)malloc( m * sizeof(int8_t));

  for( auto row=istart; row<iend; ++row ) {
  	std::memset(c_p, 0, m*sizeof(int8_t));
    std::memset(c_n, 0, m*sizeof(int8_t));

    int8x16_t accu_p4 = vdupq_n_s8(0);//vcreate_f32(0);
    int8x16_t accu_n4 = vdupq_n_s8(0);//= vcreate_f32(0);

		int num_cols_p = (int)b[row*2+0][0];
    for( auto col=1; col<=num_cols_p; ++col ) {
    	for( auto input_row=0; input_row<m/16; ++input_row ) {
      	accu_p4 = vld1q_s8(&c_p[16*input_row]);//vdupq_n_f32(0);
				int index = 16*input_row+m*(int)b[row*2+0][col];
        int8x16_t a4 = vld1q_s8(&a[index]);
        accu_p4 = vaddq_s8( a4, accu_p4 );
        vst1q_s8( &c_p[16*input_row], accu_p4 );
      }
    }

		int num_cols_n = (int)b[row*2+1][0];
		for( auto col=1; col<=num_cols_n; ++col ) {
    	for( auto input_row=0; input_row<m/16; ++input_row ) {
      	accu_n4 = vld1q_s8(&c_n[16*input_row]);//vdupq_n_f32(0);
        int index = 16*input_row+m*(int)b[row*2+1][col];
        int8x16_t a4 = vld1q_s8(&a[index]);
        accu_n4 = vaddq_s8( a4, accu_n4 );
        vst1q_s8( &c_n[16*input_row], accu_n4 );
      }
    }

    for( auto input_row=0; input_row<m/16; ++input_row ) {
    	accu_p4 = vld1q_s8(&c_p[input_row*16]);
      accu_n4 = vld1q_s8(&c_n[input_row*16]);
      int8x16_t res = vsubq_s8( accu_p4, accu_n4 );
      vst1q_s8( &c[16*input_row+m*row], res );
    }
  }
}
}
*/
