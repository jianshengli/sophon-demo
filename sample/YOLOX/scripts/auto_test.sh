#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi


usage() 
{
  echo "Usage: $0 [ -m MODE compile_nntc|compile_mlir|pcie_test|soc_build|soc_test] [ -t TARGET BM1684|BM1684X|BM1688] [ -s SOCSDK] [-a SAIL] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
}

while getopts ":m:t:s:a:d:p:" opt
do
  case $opt in 
    m)
      MODE=${OPTARG}
      echo "mode is $MODE";;
    t)
      TARGET=${OPTARG}
      echo "target is $TARGET";;
    s)
      SOCSDK=${OPTARG}
      echo "soc-sdk is $SOCSDK";;
    a)
      SAIL_PATH=${OPTARG}
      echo "sail_path is $SAIL_PATH";;
    d)
      TPUID=${OPTARG}
      echo "using tpu $TPUID";;
    p)
      PYTEST=${OPTARG}
      echo "generate logs for $PYTEST";;
    ?)
      usage
      exit 1;;
  esac
done

if test $PYTEST = "pytest"
then
  >${top_dir}auto_test_result.txt
fi

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
    if test $PYTEST = "pytest"
    then
      echo "Passed: $2" >> ${top_dir}auto_test_result.txt
      echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt
    fi
  else
    echo "Failed: $2"
    ALL_PASS=0
    if test $PYTEST = "pytest"
    then
      echo "Failed: $2" >> ${top_dir}auto_test_result.txt
      echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt
    fi
  fi

  if test $PYTEST = "pytest"
  then
    if [[ $3 != 0 ]];then
      tail -n ${ECHO_LINES} $3 >> ${top_dir}auto_test_result.txt
    fi
    echo "########Debug Info End########" >> ${top_dir}auto_test_result.txt
  fi

  sleep 3
}

function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download" 0
}

function compile_nntc()
{
  ./scripts/gen_fp32bmodel_nntc.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_int8bmodel_nntc.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
}

function compile_mlir()
{
  ./scripts/gen_fp32bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
  ./scripts/gen_fp16bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp16bmodel" 0
  ./scripts/gen_int8bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET int8bmodel" 0
}

function build_pcie()
{
  pushd cpp/yolox_$1
  if [ -d build ]; then
      rm -rf build
  fi
  mkdir build && cd build
  cmake .. && make
  judge_ret $? "build yolox_$1" 0
  popd
}

function build_soc()
{
  pushd cpp/yolox_$1
  if [ -d build ]; then
      rm -rf build
  fi
  if test $1 = "sail"; then
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK -DSAIL_PATH=$SAIL_PATH && make
    judge_ret $? "build soc yolox_$1" 0
  else
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$SOCSDK && make
    judge_ret $? "build soc yolox_$1" 0
  fi
  popd
}

function compare_res(){
    ret=`awk -v x=$1 -v y=$2 'BEGIN{print(x-y<0.002 && y-x<0.002)?1:0}'`
    if [ $ret -eq 0 ]
    then
        ALL_PASS=0
        echo "***************************************"
        echo "Ground truth is $2, your result is: $1"
        echo -e "\e[41m compare wrong! \e[0m" #red
        echo "***************************************"
        return 1
    else
        echo "***************************************"
        echo -e "\e[42m compare right! \e[0m" #green
        echo "***************************************"
        return 0
    fi
}

function test_cpp()
{
  pushd cpp/yolox_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolox_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID > log/$1_$2_$3_cpp_test.log 2>&1
  judge_ret $? "./yolox_$2.$1 --input=$4 --bmodel=../../models/$TARGET/$3 --dev_id=$TPUID" log/$1_$2_$3_cpp_test.log
  tail -n 15 log/$1_$2_$3_cpp_test.log
  if test $4 = "../../datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolox_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3
    judge_ret $? "python3 ../../tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolox_$2.$1 --language=cpp --input=log/$1_$2_$3_cpp_test.log --bmodel=$3"
    echo "==================="
  fi
  popd
}

function eval_cpp()
{
  echo -e "\n########################\nCase Start: eval cpp\n########################"
  pushd cpp/yolox_$2
  if [ ! -d log ];then
    mkdir log
  fi
  ./yolox_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1
  judge_ret $? "./yolox_$2.$1 --input=../../datasets/coco/val2017_1000 --bmodel=../../models/$TARGET/$3 --conf_thresh=0.001 --nms_thresh=0.6 --dev_id=$TPUID > log/$1_$2_$3_debug.log 2>&1" log/$1_$2_$3_debug.log
  tail -n 15 log/$1_$2_$3_debug.log
  
  echo "Evaluating..."
  res=$(python3 ../../tools/eval_coco.py --gt_path ../../datasets/coco/instances_val2017_1000.json --result_json results/$3_val2017_1000_$2_cpp_result.json 2>&1 | tee log/$1_$2_$3_eval.log)
  echo -e "$res"

  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $4
  judge_ret $? "$3_val2017_1000_$2_cpp_result: Precision compare!" log/$1_$2_$3_eval.log
  popd
  echo -e "########################\nCase End: eval cpp\n########################\n"
}

function test_python()
{
  if [ ! -d log ];then
    mkdir log
  fi
  python3 python/yolox_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID > log/$1_$2_python_test.log 2>&1
  judge_ret $? "python3 python/yolox_$1.py --input $3 --bmodel models/$TARGET/$2 --dev_id $TPUID" log/$1_$2_python_test.log
  tail -n 20 log/$1_$2_python_test.log
  if test $3 = "datasets/coco/val2017_1000"; then
    echo "==================="
    echo "Comparing statis..."
    python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolox_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2
    judge_ret $? "python3 tools/compare_statis.py --target=$TARGET --platform=${MODE%_*} --program=yolox_$1.py --language=python --input=log/$1_$2_python_test.log --bmodel=$2"
    echo "==================="
  fi
}

function eval_python()
{  
  echo -e "\n########################\nCase Start: eval python\n########################"
  if [ ! -d python/log ];then
    mkdir python/log
  fi
  python3 python/yolox_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1
  judge_ret $? "python3 python/yolox_$1.py --input datasets/coco/val2017_1000 --bmodel models/$TARGET/$2 --dev_id $TPUID --conf_thresh 0.001 --nms_thresh 0.6 > python/log/$1_$2_debug.log 2>&1" python/log/$1_$2_debug.log
  tail -n 20 python/log/$1_$2_debug.log
  
  echo "Evaluating..."
  res=$(python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/$2_val2017_1000_$1_python_result.json 2>&1 | tee python/log/$1_$2_eval.log)
  echo -e "$res"
  array=(${res//=/ })
  acc=${array[1]}
  compare_res $acc $3
  judge_ret $? "$2_val2017_1000_$1_python_result: Precision compare!" python/log/$1_$2_eval.log
  echo -e "########################\nCase End: eval python\n########################\n"
}

if test $MODE = "compile_nntc"
then
  download
  compile_nntc
elif test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  build_pcie bmcv
  build_pcie sail
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp pcie bmcv yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie sail yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie sail yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie sail yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python opencv yolox_s_fp32_1b.bmodel 0.403
    eval_python opencv yolox_s_int8_1b.bmodel 0.397
    eval_python opencv yolox_s_int8_4b.bmodel 0.397
    eval_python bmcv yolox_s_fp32_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_1b.bmodel 0.397
    eval_python bmcv yolox_s_int8_4b.bmodel 0.397
    eval_cpp pcie bmcv yolox_s_fp32_1b.bmodel 0.400
    eval_cpp pcie bmcv yolox_s_int8_1b.bmodel 0.396
    eval_cpp pcie bmcv yolox_s_int8_4b.bmodel 0.396
    eval_cpp pcie sail yolox_s_fp32_1b.bmodel 0.400
    eval_cpp pcie sail yolox_s_int8_1b.bmodel 0.396
    eval_cpp pcie sail yolox_s_int8_4b.bmodel 0.396

  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie bmcv yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp pcie sail yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp pcie bmcv yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolox_s_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie bmcv yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie sail yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie sail yolox_s_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie sail yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp pcie sail yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python opencv yolox_s_fp32_1b.bmodel 0.402
    eval_python opencv yolox_s_fp16_1b.bmodel 0.402
    eval_python opencv yolox_s_int8_1b.bmodel 0.402
    eval_python opencv yolox_s_int8_4b.bmodel 0.402
    eval_python bmcv yolox_s_fp32_1b.bmodel 0.402
    eval_python bmcv yolox_s_fp16_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_4b.bmodel 0.402
    eval_cpp pcie bmcv yolox_s_fp32_1b.bmodel 0.400
    eval_cpp pcie bmcv yolox_s_fp16_1b.bmodel 0.400
    eval_cpp pcie bmcv yolox_s_int8_1b.bmodel 0.401
    eval_cpp pcie bmcv yolox_s_int8_4b.bmodel 0.401
    eval_cpp pcie sail yolox_s_fp32_1b.bmodel 0.400
    eval_cpp pcie sail yolox_s_fp16_1b.bmodel 0.400
    eval_cpp pcie sail yolox_s_int8_1b.bmodel 0.401
    eval_cpp pcie sail yolox_s_int8_4b.bmodel 0.401
  fi
elif test $MODE = "soc_build"
then
  build_soc bmcv
  build_soc sail
elif test $MODE = "soc_test"
then
  download
  if test $TARGET = "BM1684"
  then
    test_python opencv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000

    eval_python opencv yolox_s_fp32_1b.bmodel 0.403
    eval_python opencv yolox_s_int8_1b.bmodel 0.397
    eval_python opencv yolox_s_int8_4b.bmodel 0.397
    eval_python bmcv yolox_s_fp32_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_1b.bmodel 0.397
    eval_python bmcv yolox_s_int8_4b.bmodel 0.397
    eval_cpp soc bmcv yolox_s_fp32_1b.bmodel 0.400
    eval_cpp soc bmcv yolox_s_int8_1b.bmodel 0.396
    eval_cpp soc bmcv yolox_s_int8_4b.bmodel 0.396
    eval_cpp soc sail yolox_s_fp32_1b.bmodel 0.400
    eval_cpp soc sail yolox_s_int8_1b.bmodel 0.396
    eval_cpp soc sail yolox_s_int8_4b.bmodel 0.396

  elif test $TARGET = "BM1684X"
  then
    test_python opencv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000
    
    eval_python opencv yolox_s_fp32_1b.bmodel 0.402
    eval_python opencv yolox_s_fp16_1b.bmodel 0.402
    eval_python opencv yolox_s_int8_1b.bmodel 0.402
    eval_python opencv yolox_s_int8_4b.bmodel 0.402
    eval_python bmcv yolox_s_fp32_1b.bmodel 0.402
    eval_python bmcv yolox_s_fp16_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_4b.bmodel 0.402
    eval_cpp soc bmcv yolox_s_fp32_1b.bmodel 0.400
    eval_cpp soc bmcv yolox_s_fp16_1b.bmodel 0.400
    eval_cpp soc bmcv yolox_s_int8_1b.bmodel 0.401
    eval_cpp soc bmcv yolox_s_int8_4b.bmodel 0.401
    eval_cpp soc sail yolox_s_fp32_1b.bmodel 0.400
    eval_cpp soc sail yolox_s_fp16_1b.bmodel 0.400
    eval_cpp soc sail yolox_s_int8_1b.bmodel 0.401
    eval_cpp soc sail yolox_s_int8_4b.bmodel 0.401
  elif test $TARGET = "BM1688"
  then
    test_python opencv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python opencv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/test_car_person_1080P.mp4
    test_python bmcv yolox_s_int8_4b.bmodel datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc bmcv yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolox_s_fp32_1b.bmodel ../../datasets/test_car_person_1080P.mp4
    test_cpp soc sail yolox_s_int8_4b.bmodel ../../datasets/test_car_person_1080P.mp4

    #performence test
    test_python opencv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python opencv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp32_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_fp16_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_1b.bmodel datasets/coco/val2017_1000
    test_python bmcv yolox_s_int8_4b.bmodel datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc bmcv yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_fp32_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_fp16_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_int8_1b.bmodel ../../datasets/coco/val2017_1000
    test_cpp soc sail yolox_s_int8_4b.bmodel ../../datasets/coco/val2017_1000
    
    eval_python opencv yolox_s_fp32_1b.bmodel 0.403
    eval_python opencv yolox_s_fp16_1b.bmodel 0.402
    eval_python opencv yolox_s_int8_1b.bmodel 0.402
    eval_python opencv yolox_s_int8_4b.bmodel 0.402

    eval_python bmcv yolox_s_fp32_1b.bmodel 0.402
    eval_python bmcv yolox_s_fp16_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_1b.bmodel 0.402
    eval_python bmcv yolox_s_int8_4b.bmodel 0.402

    eval_cpp soc bmcv yolox_s_fp32_1b.bmodel 0.400
    eval_cpp soc bmcv yolox_s_fp16_1b.bmodel 0.400
    eval_cpp soc bmcv yolox_s_int8_1b.bmodel 0.402
    eval_cpp soc bmcv yolox_s_int8_4b.bmodel 0.402

    eval_cpp soc sail yolox_s_fp32_1b.bmodel 0.400
    eval_cpp soc sail yolox_s_fp16_1b.bmodel 0.400
    eval_cpp soc sail yolox_s_int8_1b.bmodel 0.402
    eval_cpp soc sail yolox_s_int8_4b.bmodel 0.402
  
    eval_python opencv yolox_s_fp32_1b_2core.bmodel 0.403
    eval_python opencv yolox_s_fp16_1b_2core.bmodel 0.402
    eval_python opencv yolox_s_int8_1b_2core.bmodel 0.402
    eval_python opencv yolox_s_int8_4b_2core.bmodel 0.402

    eval_python bmcv yolox_s_fp32_1b_2core.bmodel 0.402
    eval_python bmcv yolox_s_fp16_1b_2core.bmodel 0.402
    eval_python bmcv yolox_s_int8_1b_2core.bmodel 0.402
    eval_python bmcv yolox_s_int8_4b_2core.bmodel 0.402

    eval_cpp soc bmcv yolox_s_fp32_1b_2core.bmodel 0.400
    eval_cpp soc bmcv yolox_s_fp16_1b_2core.bmodel 0.400
    eval_cpp soc bmcv yolox_s_int8_1b_2core.bmodel 0.402
    eval_cpp soc bmcv yolox_s_int8_4b_2core.bmodel 0.402

    eval_cpp soc sail yolox_s_fp32_1b_2core.bmodel 0.400
    eval_cpp soc sail yolox_s_fp16_1b_2core.bmodel 0.400
    eval_cpp soc sail yolox_s_int8_1b_2core.bmodel 0.402
    eval_cpp soc sail yolox_s_int8_4b_2core.bmodel 0.402
  fi
fi

if [[ $ALL_PASS -eq 0 ]]
then
    echo "===================================================================="
    echo "Some process produced unexpected results, please look out their logs!"
    echo "===================================================================="
else
    echo "===================="
    echo "Test cases all pass!"
    echo "===================="
fi

popd