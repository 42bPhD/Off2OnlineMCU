
// #include <Arduino.h>
// #include "arm_nnsupportfunctions.h"
// #include "arm_math.h"
// #include <arm_nnfunctions.h>
// #include <stdlib.h>
// // #include "inputs.h"
// // #include "parameter.h"
// // #include "weights.h"
// // #include "modules.h"
// // #include "memory_pack.h"
// #include "nnom.h"
// #include "weights.h"


// void reset_cnt()
// {
//     CoreDebug->DEMCR |= 0x01000000;
//     DWT->CYCCNT = 0; // reset the counter
//     DWT->CTRL = 0;
// }

// void start_cnt()
// {
//     DWT->CTRL |= 0x00000001; // enable the counter
// }

// void stop_cnt()
// {
//     DWT->CTRL &= 0xFFFFFFFE; // disable the counter
// }

// unsigned int getCycles()
// {
//     return DWT->CYCCNT;
// }
// #if !defined DWT_LSR_Present_Msk
// #define DWT_LSR_Present_Msk ITM_LSR_Present_Msk
// #endif
// #if !defined DWT_LSR_Access_Msk
// #define DWT_LSR_Access_Msk ITM_LSR_Access_Msk
// #endif
// #define DWT_LAR_KEY 0xC5ACCE55
// void dwt_access_enable(unsigned ena)
// {
//     uint32_t lsr = DWT->LSR;
//     ;

//     if ((lsr & DWT_LSR_Present_Msk) != 0)
//     {
//         if (ena)
//         {
//             if ((lsr & DWT_LSR_Access_Msk) != 0) // locked: access need unlock
//             {
//                 DWT->LAR = DWT_LAR_KEY;
//             }
//         }
//         else
//         {
//             if ((lsr & DWT_LSR_Access_Msk) == 0) // unlocked
//             {
//                 DWT->LAR = 0;
//             }
//         }
//     }
// }

// //uint8_t static_buf[1024 * 3];
// nnom_predict_t * pre;
// uint32_t label=0;
// int8_t* input;
// size_t size = 0;
// float prob;
// nnom_model_t *model = nnom_model_create();

// void setup() {
//     Serial.begin(115200);
//     SCB_EnableICache();
//     SCB_EnableDCache();
//     //nnom_set_static_buf(static_buf, sizeof(static_buf)); // set static memory block for nnom
//     model_run(model);
// }
// q7_t TEST_DATA[28*28] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,92,79,75,30,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,111,127,127,127,127,120,99,99,99,99,99,99,99,99,85,26,0,0,0,0,0,0,0,0,0,0,0,0,33,57,36,57,81,113,127,112,127,127,127,125,114,127,127,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,33,7,33,33,33,29,10,118,127,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,126,104,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,116,127,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,127,119,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,124,127,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,127,93,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,102,124,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,127,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,125,120,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,110,127,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,101,127,109,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,127,127,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,112,127,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,127,127,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,121,127,127,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,127,127,109,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,127,103,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
// void loop() {
    

//     nnom_predict(model, &label, &prob);
    
// 	// model
// 	model_stat(model);

//     Serial.println("label: " + String(label) + " prob: " + String(prob));
//     delay(3000);
    
// }   


