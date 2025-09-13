import {Router} from "express";
import {login, register, verifyOtp,resendOtp} from "../controllers/User.controller.js"
import {checkRole} from "../middlewares/checkRole.js";
import { protectRoute } from "../middlewares/protectRoute.js";


const router = Router();

router.route("/login").post(login);
router.route("/register").post(register);
router.route("/verify-otp").post(verifyOtp);
// router.route('/logout').post(logout);
router.route("/resendotp").post(resendOtp);

export default router;
