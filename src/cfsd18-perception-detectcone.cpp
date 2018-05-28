/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "detectcone.hpp"

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{0};
    std::map<std::string, std::string> commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if (commandlineArguments.count("cid")<1) {
        std::cerr << argv[0] << " is a detectcone module for the CFSD18 project." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OpenDaVINCI session> [--id=<Identifier in case of simulated units>] [--verbose] [Module specific parameters....]" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=111 --id=120"  <<  std::endl;
        retCode = 1;
    } 
    else {
//         bool const VERBOSE{commandlineArguments.count("verbose") != 0};
//         (void)VERBOSE;
//         // Interface to a running OpenDaVINCI session (ignoring any incoming Envelopes).
//         cluon::data::Envelope data;
//         //std::shared_ptr<Slam> slammer = std::shared_ptr<Slam>(new Slam(10));
//         cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};
//         DetectCone detectcone(commandlineArguments,od4);

//         cv::Mat img = cv::imread("0.png");
//         detectcone.forwardDetectionORB(img);


//         auto envelopeRecieved{[&logic = detectcone](cluon::data::Envelope &&envelope)
//           {
//             logic.nextContainer(envelope);
//           } };

//         od4.dataTrigger(opendlv::logic::perception::ObjectDirection::ID(),envelopeRecieved);
//         od4.dataTrigger(opendlv::logic::perception::ObjectDistance::ID(),envelopeRecieved);

//         // Just sleep as this microservice is data driven.
//         using namespace std::literals::chrono_literals;
//         while (od4.isRunning()) {
//           std::this_thread::sleep_for(1s);
//           std::chrono::system_clock::time_point tp;
//         }
//     }
//     return retCode;
// }



    const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
    const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
    const uint32_t BPP{static_cast<uint32_t>(std::stoi(commandlineArguments["bpp"]))};

    if ( (BPP != 24) && (BPP != 8) ) {
        std::cerr << argv[0] << ": bits per pixel must be either 24 or 8; found " << BPP << "." << std::endl;
    }
    else {
        const uint32_t SIZE{WIDTH * HEIGHT * BPP/8};
        const std::string NAME{(commandlineArguments["name"].size() != 0) ? commandlineArguments["name"] : "/camera1"};
        const uint32_t ID{(commandlineArguments["id"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["id"])) : 0};
        const bool VERBOSE{commandlineArguments.count("verbose") != 0};

        (void)ID;
        (void)SIZE;
        std::cout << "Making slammer" << VERBOSE << std::endl;

        // Interface to a running OpenDaVINCI session (ignoring any incoming Envelopes).
        cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

        DetectCone detectcone(commandlineArguments, od4);
        
        size_t frameCounter = 0;
        
        std::unique_ptr<cluon::SharedMemory> sharedMemory(new cluon::SharedMemory{NAME});
        if (sharedMemory && sharedMemory->valid()) {
            std::clog << argv[0] << ": Found shared memory '" << sharedMemory->name() << "' (" << sharedMemory->size() << " bytes)." << std::endl;

            cv::Size size;
            size.width = WIDTH;
            size.height = HEIGHT;

            IplImage *image = cvCreateImageHeader(size, IPL_DEPTH_8U, BPP/8);
            sharedMemory->lock();
            image->imageData = sharedMemory->data();
            image->imageDataOrigin = image->imageData;
            sharedMemory->unlock();
            //size_t lastMapPoint = 0;
            //uint32_t lastSentIndex = 0;
            while (od4.isRunning()) {
                // The shared memory uses a pthread broadcast to notify us; just sleep to get awaken up.
                
                sharedMemory->wait();
                
                sharedMemory->lock();
                image->imageData = sharedMemory->data();
                image->imageDataOrigin = image->imageData;
                cv::Mat img = cv::cvarrToMat(image); 
                
                sharedMemory->unlock();
                cv::waitKey(1);
                
                // cv::namedWindow("img", cv::WINDOW_NORMAL);
                // cv::imshow("img", img);
                // cv::waitKey(2);
                // cv::imwrite("test.png",img);

                // cv::Mat img2 = cv::imread("0.png");
                detectcone.forwardDetectionORB(img);
                frameCounter++;
            }

            cvReleaseImageHeader(&image);
        }
        else {
            std::cerr << argv[0] << ": Failed to access shared memory '" << NAME << "'." << std::endl;
        }
      }

  }
  return retCode;
}









