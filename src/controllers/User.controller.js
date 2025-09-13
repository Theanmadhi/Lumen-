import { User } from "../models/userModel.js";
import httpStatus from "http-status";
import bcrypt from "bcrypt";
import crypto from "crypto";
import nodemailer from "nodemailer";
import dns from "dns/promises";
import jwt from "jsonwebtoken";

function isValidEmail(email) {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
}

async function canReceiveEmail(email) {
  const domain = email.split("@")[1];
  try {
    const records = await dns.resolveMx(domain);
    return records && records.length > 0;
  } catch {
    return false;
  }
}

const login = async (req,res)=>{
    const {username,password} = req.body;
    if(!username || !password){
        return res.status(400).json({message: "Enter all the credentials"});
    }
    try{
        const user = await User.findOne({username});
        if(!user){
            return res.status(httpStatus.NOT_FOUND).json({message:"User Not Found"});
        }
        if(!user.isVerified){
            return res.status(httpStatus.UNAUTHORIZED).json({message:"Please verify your email first"});
        }
        const isMatch = await bcrypt.compare(password, user.password);
        if(!isMatch){
            return res.status(httpStatus.UNAUTHORIZED).json({message:"Wrong Password"});
        }
        let token = jwt.sign({id: user._id},process.env.JWT_SECRET || "mysecretkey",{
            expiresIn:"1d",
        });
        return res.status(httpStatus.OK).json({token: token});
    }catch(e){
        return res.status(500).json({message:`Something went wrong in login  ${e}`});
    }
}



const register = async (req, res) => {
  const { username, email, password } = req.body;

  if (!username || !email || !password) {
    return res.status(httpStatus.BAD_REQUEST).json({ message: "All fields required" });
  }

  if (!isValidEmail(email) || !(await canReceiveEmail(email))) {
    return res.status(httpStatus.BAD_REQUEST).json({ message: "Invalid or non-existent email" });
  }

  try {
    const existingUser = await User.findOne({ $or: [{ username }, { email }] });
    if (existingUser) {
      return res.status(httpStatus.FOUND).json({ message: "Username or email already exists" });
    }
    const hashedPassword = await bcrypt.hash(password, 10);
    const otp = crypto.randomInt(100000, 999999).toString();
    const otpExpires = new Date(Date.now() + 10 * 60 * 1000); // 10 min
    const newUser = new User({
      username,
      email,
      password: hashedPassword,
      otp,
      otpExpires,
      isVerified: false,
    });

    await newUser.save();
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.GMAIL_USER,
        pass: process.env.GMAIL_PASS, 
      },
    });

    const mailOptions = {
      from: process.env.GMAIL_USER,
      to: email,
      subject: "Verify your email",
      text: `Your OTP is: ${otp}. It will expire in 10 minutes.`,
    };

    await transporter.sendMail(mailOptions);

    res.status(httpStatus.CREATED).json({
      message: "User registered. Please verify OTP sent to your email.",
    });
  } catch (e) {
    console.error("Register error:", e);
    res.status(500).json({ message: `Something went wrong in register: ${e}` });
  }
};
const verifyOtp = async (req, res) => {
  const { email, otp } = req.body;

  if (!email || !otp) {
    return res.status(httpStatus.BAD_REQUEST).json({ message: "Email and OTP required" });
  }

  try {
    const user = await User.findOne({ email });

    if (!user) return res.status(httpStatus.NOT_FOUND).json({ message: "User not found" });
    if (user.isVerified) return res.status(httpStatus.BAD_REQUEST).json({ message: "User already verified" });
    if (user.otp !== otp || user.otpExpires < new Date()) {
      return res.status(httpStatus.BAD_REQUEST).json({ message: "Invalid or expired OTP" });
    }

    user.isVerified = true;
    user.otp = null;
    user.otpExpires = null;

    await user.save();

    res.status(httpStatus.OK).json({ message: "Email verified successfully" });
  } catch (e) {
    res.status(500).json({ message: `Something went wrong in OTP verification ${e}` });
  }
};


const resendOtp = async (req, res) => {
  const { email } = req.body;

  const user = await User.findOne({ email });
  if (!user) return res.status(404).json({ error: "User not found" });
  if (user.isVerified) return res.status(400).json({ error: "User already verified" });

  const otp = crypto.randomInt(100000, 999999).toString();
  const otpExpires = new Date(Date.now() + 10 * 60 * 1000);

  user.otp = otp;
  user.otpExpires = otpExpires;
  await user.save();
  const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.GMAIL_USER,
        pass: process.env.GMAIL_PASS, 
      },
    });

    const mailOptions = {
      from: process.env.GMAIL_USER,
      to: email,
      subject: "Verify your email",
      text: `Your OTP is: ${otp}. It will expire in 10 minutes.`,
    };

    await transporter.sendMail(mailOptions);

  res.json({ message: "OTP sent successfully" });
};

export {login,register, verifyOtp,resendOtp};